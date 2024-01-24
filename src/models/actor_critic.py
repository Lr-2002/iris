import random
from dataclasses import dataclass
from typing import Any, Optional, Union
import sys

from einops import rearrange
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import Batch
from envs.world_model_env import WorldModelEnv
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import compute_lambda_returns, LossWithIntermediateLosses


@dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor
    means_values: torch.FloatTensor


@dataclass
class ImagineOutput:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    logits_actions: torch.FloatTensor
    values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor


class ActorCritic(nn.Module):
    def __init__(self, act_vocab_size, ac_env, use_original_obs: bool = False) -> None:
        super().__init__()
        self.use_original_obs = use_original_obs
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        mlp_size = 512
        self.fc1 = nn.Linear(1, mlp_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mlp_size, mlp_size)
        self.fc3 = nn.Linear(mlp_size, mlp_size)
        self.fc4 = nn.Linear(mlp_size, 1)


        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(1025, self.lstm_dim)
        self.hx, self.cx = None, None

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, act_vocab_size)

        self.ac_env  = ac_env

    def __repr__(self) -> str:
        return "actor_critic"

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None) -> None:
        device = self.conv1.weight.device
        self.hx = torch.zeros(n, self.lstm_dim, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, device=device)
        if burnin_observations is not None:
            image = burnin_observations['image']
            token = burnin_observations['token']
            assert image.ndim == 5 and image.size(0) == n and mask_padding is not None and image.shape[:2] == mask_padding.shape
            for i in range(image.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self({'image':image[:, i], 'token':token[:,i]}, mask_padding[:, i])

    def prune(self, mask: np.ndarray) -> None:
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    def forward(self, inputs: dict, mask_padding: Optional[torch.BoolTensor] = None) -> ActorCriticOutput:
        assert 'image' in inputs.keys() and 'token' in inputs.keys() , f"inputs is {inputs}"
        image = inputs['image']
        token = inputs['token']
        assert token.ndim == 2 and token.shape[-1] == 1, token.shape
        assert image.ndim == 4 and image.shape[1:] == (3, 64, 64) , image.shape
        assert 0 <= image.min() <= 1 and 0 <= image.max() <= 1
        assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.size(0) == image.size(0) and mask_padding.any())
        x = image[mask_padding] if mask_padding is not None else image
        token = token[mask_padding] if mask_padding is not None else token

        x = x.mul(2).sub(1)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        # print('in ac x_shape is ', x.shape)
        token = token.to(torch.float32)
        token = self.relu(self.fc1(token))
        token = self.relu(self.fc2(token))
        token = self.relu(self.fc3(token))
        token = self.relu(self.fc4(token))
        x = torch.cat((x, token), dim=1)

        if mask_padding is None:
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))

        logits_actions = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
        means_values = rearrange(self.critic_linear(self.hx), 'b 1 -> b 1 1')

        return ActorCriticOutput(logits_actions, means_values)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, imagine_horizon: int, gamma: float, lambda_: float, entropy_weight: float, **kwargs: Any) -> LossWithIntermediateLosses:
        assert not self.use_original_obs
        # outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)
        outputs = self.rollout_in_env(batch, horizon=imagine_horizon)
        # for i in len(imagine_horizon):

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs.values[:, :-1]

        d = Categorical(logits=outputs.logits_actions[:, :-1])
        log_probs = d.log_prob(outputs.actions[:, :-1])
        loss_actions = -1 * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = - entropy_weight * d.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns)
        # total_reward = outputs.rewards.sum(dim=1).float().mean()

        return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy)

    def rollout_in_env(self, batch: Batch, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        # use the multi env to get the data and train it on the env
        # the batch is not usefull, and i'll start the env in reset
        # and then burn them in the model
        # when the steps have got to 20 ,then start the collection time
        device = batch['observations']['token'].device
        obs= self.ac_env.reset()
        obs = {'image': rearrange(torch.tensor(obs['image']).to(device) / 255, 'n w h c -> n c w h'),
               'token': torch.tensor(obs['token']).unsqueeze(1).to(device)}

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []
        self.reset(n=self.ac_env.num_envs)

        for k in tqdm(range(horizon), disable=not show_pbar, desc='Rollout', file=sys.stdout):
            # imagine todo
            all_observations.append(obs)

            outputs_ac = self(obs)
            action_token = (Categorical(logits=outputs_ac.logits_actions).sample())
            act= action_token.squeeze(-1).cpu().numpy()
            obs, reward, done, _ = self.ac_env.step(act)
            obs = {'image': rearrange(torch.tensor(obs['image']).to(device) / 255, 'n w h c -> n c w h'),
                   'token': torch.tensor(obs['token']).unsqueeze(1).to(device)}
            reward = torch.tensor(reward).unsqueeze(1).to(device)
            done = torch.tensor(done).unsqueeze(1).to(device)
            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))
        self.clear()
        # print('all obs ', all_observations)
        imgs = [x['image'] for x in all_observations]
        imgs = torch.stack(imgs, dim=1).mul(255).byte()
        tokens = [x['token'] for x in all_observations]
        tokens = torch.cat(tokens, dim=1)
        return ImagineOutput(
            observations={'image':imgs, 'token':tokens},      # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )

    def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        assert not self.use_original_obs
        initial_observations = batch['observations']['image']
        tokens = batch['observations']['token']
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = initial_observations.device
        wm_env = WorldModelEnv(tokenizer, world_model, device)
        # real_env =
        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        burnin_observations = {'image': torch.clamp(tokenizer.encode_decode(initial_observations[:, :-1], should_preprocess=True, should_postprocess=True), 0, 1) if initial_observations.size(1) > 1 else None ,'token':rearrange( tokens[:,-1].unsqueeze(1).expand(-1,20).unsqueeze(1), "a b c -> a c b") if initial_observations.size(1) > 1 else None}
        self.reset(n=initial_observations.size(0), burnin_observations=burnin_observations, mask_padding=mask_padding[:, :-1])
        obs_img = initial_observations[:, -1]
        obs_tok = batch['observations']['token'][:,-1].unsqueeze(1)
        obs = wm_env.reset_from_initial_observations({'image':obs_img, 'token':obs_tok})
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):
            # imagine todo
            all_observations.append(obs)

            outputs_ac = self(obs)
            action_token = Categorical(logits=outputs_ac.logits_actions).sample()
            obs, reward, done, _ = wm_env.step(action_token, should_predict_next_obs=(k < horizon - 1))
            # print('show obs token', obs['token'])
            if obs is not None:
                assert obs['token'].shape == obs_tok.shape
                # task = torch.tensor([0] * len(reward), device=0)
                obs['token'] = obs_tok
                # print('before random ', obs['token'])
                for tk in range(len(reward)):
                    if reward[tk] == 1:
                        obs['token'][tk] = random.randint(0,37)
                # print('show obs token', obs['token'])
                # print('show reward ', reward)


            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()
        # print('all obs ', all_observations)
        imgs = [x['image'] for x in all_observations]
        imgs = torch.stack(imgs, dim=1).mul(255).byte()
        tokens = [x['token'] for x in all_observations]
        tokens = torch.cat(tokens, dim=1)
        return ImagineOutput(
            observations={'image':imgs, 'token':tokens},      # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )
