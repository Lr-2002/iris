import random
import sys
from typing import List, Optional, Union

from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm
import wandb

from agent import Agent
from dataset import EpisodesDataset
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from utils import EpisodeDirManager, RandomHeuristic


class Collector:
    def __init__(self, env: Union[SingleProcessEnv, MultiProcessEnv], dataset: EpisodesDataset, episode_dir_manager: EpisodeDirManager) -> None:
        self.env = env
        self.dataset = dataset
        self.episode_dir_manager = episode_dir_manager
        self.obs = self.env.reset()
        self.episode_ids = [None] * self.env.num_envs
        self.heuristic = RandomHeuristic(self.env.num_actions)

    @torch.no_grad()
    def collect(self, agent: Agent, epoch: int, epsilon: float, should_sample: bool, temperature: float, burn_in: int, *, num_steps: Optional[int] = None, num_episodes: Optional[int] = None, info_flag = False):
        assert self.env.num_actions == agent.world_model.act_vocab_size
        assert 0 <= epsilon <= 1

        assert (num_steps is None) != (num_episodes is None)
        # if info_flag == True:
        #     print('now find it ')
        should_stop = lambda steps, episodes: steps >= num_steps if num_steps is not None else episodes >= num_episodes

        to_log = []
        steps, episodes = 0, 0
        returns = []
        observations, actions, rewards, dones = [], [], [], []

        burnin_obs_rec, mask_padding = None, None
        # print("burnin", burn_in)
        if set(self.episode_ids) != {None} and burn_in > 0:
            current_episodes = [self.dataset.get_episode(episode_id) for episode_id in self.episode_ids]
            segmented_episodes = [episode.segment(start=len(episode) - burn_in, stop=len(episode), should_pad=True) for episode in current_episodes]
            mask_padding = torch.stack([episode.mask_padding for episode in segmented_episodes], dim=0).to(agent.device)
            # burnin_obs = torch.stack([episode.observations['image'] for episode in segmented_episodes], dim=0).float().div(255).to(agent.device)
            burnin_obs_image = torch.stack([episode.observations['image'] for episode in segmented_episodes],
                                           dim=0).float().div(255).to(agent.device)
            burnin_obs_token = torch.stack([episode.observations['token'] for episode in segmented_episodes], dim=0).to(
                agent.device)
            burnin_obs = {'image': burnin_obs_image, 'token': burnin_obs_token}
            # print("token shape is ", burnin_obs_token.shape)
            # print('token in collector is ', burnin_obs_token)
            burnin_obs_rec = {'image':torch.clamp(agent.tokenizer.encode_decode(burnin_obs, should_preprocess=True, should_postprocess=True), 0, 1) , 'token':rearrange(burnin_obs_token.unsqueeze(1), 'a b c -> a c b')}

        agent.actor_critic.reset(n=self.env.num_envs, burnin_observations=burnin_obs_rec, mask_padding=mask_padding)
        pbar = tqdm(total=num_steps if num_steps is not None else num_episodes, desc=f'Experience collection ({self.dataset.name})', file=sys.stdout)
        # print('total ', num_steps , num_episodes)
        while not should_stop(steps, episodes):

            observations.append(self.obs)
            # print('in collector self.obs', self.obs.shape)
            # print('bug position :', self.obs)
            # if info_flag :
                 # print()
            img = rearrange(torch.FloatTensor(self.obs['image']).div(255), 'n h w c -> n c h w').to(agent.device)
            token = torch.LongTensor(self.obs['token']).unsqueeze(-1).to(agent.device)
            obs = {'image':img, 'token':token}
            act = agent.act(obs, should_sample=should_sample, temperature=temperature).cpu().numpy()

            if random.random() < epsilon:
                act = self.heuristic.act(obs).cpu().numpy()

            self.obs, reward, done, _ = self.env.step(act)
            # print('self.obs is ', self.obs)
            # print(self.env.unwrapped_)
            # print('token: ', self.env.env.unwrapped.original_token)
            # print(dir(self.env.env.env.env.unwrapped))
            actions.append(act)
            rewards.append(reward)
            dones.append(done)
            # print(done)
            new_steps = len(self.env.mask_new_dones)
            steps += new_steps
            # print(steps, new_steps)
            pbar.update(new_steps if num_steps is not None else 0)

            # Warning: with EpisodicLifeEnv + MultiProcessEnv, reset is ignored if not a real done.
            # Thus, segments of experience following a life loss and preceding a general done are discarded.
            # Not a problem with a SingleProcessEnv.
            # print('self.env.should_reset', self.env.should_reset())
            if self.env.should_reset():
                self.add_experience_to_dataset(observations, actions, rewards, dones)

                new_episodes = self.env.num_envs
                episodes += new_episodes
                pbar.update(new_episodes if num_episodes is not None else 0)

                for episode_id in self.episode_ids:
                    episode = self.dataset.get_episode(episode_id)
                    self.episode_dir_manager.save(episode, episode_id, epoch)
                    metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
                    metrics_episode['episode_num'] = episode_id
                    metrics_episode['action_histogram'] = wandb.Histogram(np_histogram=np.histogram(episode.actions.numpy(), bins=np.arange(0, self.env.num_actions + 1) - 0.5, density=True))
                    to_log.append({f'{self.dataset.name}/{k}': v for k, v in metrics_episode.items()})
                    returns.append(metrics_episode['episode_return'])

                self.obs = self.env.reset()
                self.episode_ids = [None] * self.env.num_envs
                agent.actor_critic.reset(n=self.env.num_envs)
                observations, actions, rewards, dones = [], [], [], []

        # Add incomplete episodes to dataset, and complete them later.
        if len(observations) > 0:
            self.add_experience_to_dataset(observations, actions, rewards, dones)

        agent.actor_critic.clear()

        metrics_collect = {
            '#episodes': len(self.dataset),
            '#steps': sum(map(len, self.dataset.episodes)),
        }
        if len(returns) > 0:
            metrics_collect['return'] = np.mean(returns)
        metrics_collect = {f'{self.dataset.name}/{k}': v for k, v in metrics_collect.items()}
        to_log.append(metrics_collect)

        return to_log

    def add_experience_to_dataset(self, observations: List[dict], actions: List[np.ndarray], rewards: List[np.ndarray], dones: List[np.ndarray]) -> None:
        # print('---------------', len(observations), len(actions) , len(rewards), len(dones))
        assert len(observations) == len(actions) == len(rewards) == len(dones)
        # print('len observations', len(observations))
        # for i, (o, a, r, d) in enumerate(zip(*map(lambda arr: np.swapaxes(arr, 0, 1), [observations, actions, rewards, dones]))):  # Make everything (N, T, ...) instead of (T, N, ...)
        # for i, (o, a, r, d) in enumerate(zip(*map(lambda arr: np.swapaxes(arr, 0, 1), [observations, actions, rewards, dones]))):  # Make everything (N, T, ...) instead of (T, N, ...)
        # tt = []
        # for i in observations:
        #     obs.append({'image': np.swapaxes(i['image'], 0, 1), 'token': i['token']})
        obs_img = [i['image'] for i in observations]
        obs_token = [i['token'] for i in observations]  # todo this might be wrong
        # print('obs_token is ', obs_token)
        # obs_img = np.swapaxes(obs_img, 0, 1)
        # obs_token = np.swapaxes(obs_token, 0, 1)
        obs_img = np.array(obs_img)
        obs_token = np.array(obs_token)
        tt = [*map(lambda arr: np.swapaxes(arr, 0, 1), [obs_img, obs_token, actions, rewards, dones])]
        # print('tt', tt )
        for i, (img ,tok, a, r, d) in enumerate(zip(*tt)):
            # print('iter', i)
            img = torch.ByteTensor(img).permute(0, 3, 1, 2).contiguous()
            tok = torch.LongTensor(tok)


            episode = Episode(
                observations={'image':img, 'token':tok},  # channel-first
                actions=torch.LongTensor(a),
                rewards=torch.FloatTensor(r),
                ends=torch.LongTensor(d),
                mask_padding=torch.ones(d.shape[0], dtype=torch.bool),
            )
            # print(self.episode_ids)
            if self.episode_ids[i] is None:
                self.episode_ids[i] = self.dataset.add_episode(episode)
            else:
                self.dataset.update_episode(self.episode_ids[i], episode)
