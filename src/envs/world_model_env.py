import random
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision


class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        # print(self.env.reset()['image'])
        data = self.env.reset()
        obs_img = data['image']
        obs_tok = [data['token']]
        obs_img = torchvision.transforms.functional.to_tensor(obs_img).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        # obs_tok = torchvision.transforms.functional.to_tensor(obs_tok).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        print('obs_img shape',  obs_img.shape)
        obs_tok = torch.tensor(obs_tok).to(self.device).unsqueeze(0)
        print('obs_tok shape', obs_tok.shape )
        obs = {'image': obs_img, 'token': obs_tok}
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_img = observations['image']
        obs_tok = observations['token']
        obs_tokens = self.tokenizer.encode(obs_img, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)  64， 16
        # print('before cat token ', obs_tokens.shape) # todo token 从哪里传进来
        obs_tokens = torch.cat((obs_tokens, obs_tok),dim=1)
        # print('after cat token ', obs_tokens.shape)
        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens # 17

        _ = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens)
        self.obs_tokens = obs_tokens

        return self.decode_obs_tokens()

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:

        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens, f"num_obs {num_observations_tokens}, self.num_obs {self.num_observations_tokens}"
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True) -> None:
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None
        # todo 写一下embedder怎么处理
        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []
        # self.obs_tokens = torch.cat((self.obs_tokens, torch.zeros((self.obs_tokens.shape[0] , 1)).to(self.device)),dim=1)
        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        for k in range(num_passes):  # assumption that there is only one action token.
            # todo 忽略了最后一个task token 然后把任务里的最后一个token取出来给进去

            outputs_wm = self.world_model(token, past_keys_values=self.keys_values_wm) # 64 1
            output_sequence.append(outputs_wm.output_sequence)

            if k == 0:
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) / 2   # (B,)
                # print('world model env reward', reward)
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

            if k < self.num_observations_tokens :
                if k == self.num_observations_tokens - 1:
                    token = torch.zeros_like(token)
                    obs_tokens.append(token)
                    continue
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)        # (B, K)

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        # print('obs shape is ', obs.shape)
        return obs, reward, done, None

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        # todo get decode the word
        q = self.obs_tokens[:, :-1]
        # print('the word token is ', self.obs_tokens[:, -1])
        embedded_tokens = self.tokenizer.embedding(q)     # (B, K, E)
        # embedded_tokens = self.tokenizer.embedding(self.obs_tokens)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return {'image':torch.clamp(rec, 0, 1),'token':self.obs_tokens[:,-1].unsqueeze(-1)}

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
