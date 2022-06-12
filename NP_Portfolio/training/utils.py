import copy
import itertools

import numpy as np

from data import data_loading
from environment import portfolio_env


def load_env(env_config, env_cls=None):
    history, start_idx = data_loading.get_data(env_config['data_loader'], env_config['data_config'])
    stocks = copy.deepcopy(env_config['data_config']['stocks'])
    kwargs = copy.deepcopy(env_config['base_env_params'])
    kwargs['start_idx'] = start_idx
    if env_cls is None:
        return portfolio_env.PortfolioEnv(data=(history,stocks), **kwargs)
    else:
        return env_cls(data=(history,stocks), **kwargs)


def rollout(env, policy, num_episodes=1):
    rewards = []
    actions = []
    for _ in range(num_episodes):
        rewards.append(0)
        obs = np.expand_dims(env.reset(), 0)
        done = False
        while not done:
            action = policy.evaluate(obs).flatten()
            next_obs, reward, done, info = env.step(action)
            obs = np.expand_dims(next_obs, 0)
            rewards[-1] += reward
            actions.append(action)
    return rewards, actions


def get_rewards_stats(rewards):
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) if len(rewards) > 1 else 0.0 
    q25_reward = np.quantile(rewards, 0.25) if len(rewards) > 1 else mean_reward
    q50_reward = np.quantile(rewards, 0.50) if len(rewards) > 1 else mean_reward
    q75_reward = np.quantile(rewards, 0.75) if len(rewards) > 1 else mean_reward
    return mean_reward, std_reward, q25_reward, q50_reward, q75_reward


def get_actions_stats(actions):
    mean_action = np.mean(np.array(actions), axis=0)
    std_action = np.std(np.array(actions), axis=0)
    q25_action = np.quantile(np.array(actions), q=0.25, axis=0)
    q50_action = np.quantile(np.array(actions), q=0.50, axis=0)
    q75_action = np.quantile(np.array(actions), q=0.75, axis=0)
    return mean_action, std_action, q25_action, q50_action, q75_action


class SingleEnvEvaluator:

    def __init__(self, env, num_episodes, data_freq='1D'):
        self.env = env
        self.num_episodes = num_episodes
        if data_freq == '1D':
            self.steps_per_year = 253
        elif data_freq == '30m':
            self.steps_per_year = 48 * 253 
        else:
            raise

    def build_header(self):
        return (
            [
                'steps',
                'mean_episode_reward',
                'std_episode_reward',
                'q25_episode_reward', 
                'q50_episode_reward', 
                'q75_episode_reward', 
                'mean_episode_return',
                'std_episode_return',
                'q25_episode_return',
                'q50_episode_return',
                'q75_episode_return',
                'annualised_mean_episode_return',
                'annualised_std_episode_return',
                'annualised_q25_episode_return',
                'annualised_q50_episode_return',
                'annualised_q75_episode_return',
            ] + 
            ['mean_{}'.format(e) for e in (['CASH'] + self.env.abbreviation)] +
            ['std_{}'.format(e) for e in (['CASH'] + self.env.abbreviation)] +
            ['q25_{}'.format(e) for e in (['CASH'] + self.env.abbreviation)] +
            ['q50_{}'.format(e) for e in (['CASH'] + self.env.abbreviation)] +
            ['q75_{}'.format(e) for e in (['CASH'] + self.env.abbreviation)])

    def build_row(self, steps, policy):
        row = [str(steps)]
        rewards, actions = rollout(self.env, policy, num_episodes=self.num_episodes)
        mean_reward, std_reward, q25_reward, q50_reward, q75_reward = \
            get_rewards_stats(rewards)
        row.extend([
            '{:6f}'.format(mean_reward),
            '{:6f}'.format(std_reward),
            '{:6f}'.format(q25_reward),
            '{:6f}'.format(q50_reward),
            '{:6f}'.format(q75_reward)
        ])
        mean_return, std_return, q25_return, q50_return, q75_return = \
            get_rewards_stats([np.exp(v) for v in rewards])
        row.extend([
            '{:6f}'.format(mean_return),
            '{:6f}'.format(std_return),
            '{:6f}'.format(q25_return),
            '{:6f}'.format(q50_return),
            '{:6f}'.format(q75_return)
        ])
        annualized_mean_return, annualized_std_return, annualized_q25_return, annualized_q50_return, annualized_q75_return = \
            get_rewards_stats([np.exp(v) ** (self.steps_per_year / self.env.steps) for v in rewards])
        row.extend([
            '{:6f}'.format(annualized_mean_return),
            '{:6f}'.format(annualized_std_return),
            '{:6f}'.format(annualized_q25_return),
            '{:6f}'.format(annualized_q50_return),
            '{:6f}'.format(annualized_q75_return)
        ])
        mean_action, std_action, q25_action, q50_action, q75_action = \
            get_actions_stats(actions)
        row.extend(['{:6f}'.format(e) for e in mean_action])
        row.extend(['{:6f}'.format(e) for e in std_action])
        row.extend(['{:6f}'.format(e) for e in q25_action])
        row.extend(['{:6f}'.format(e) for e in q50_action])
        row.extend(['{:6f}'.format(e) for e in q75_action])
        return row


class MultiEnvEvaluator:

    def __init__(self, envs, num_episodes, data_freq='1D'):
        self.envs = envs
        self.num_episodes = num_episodes
        if data_freq == '1D':
            self.steps_per_year = 253
        elif data_freq == '30m':
            self.steps_per_year = 48 * 253 
        else:
            raise

    def build_header(self):
        return (
            [
                'steps',
                'mean_episode_reward',
                'std_episode_reward', 
                'q25_episode_reward', 
                'q50_episode_reward', 
                'q75_episode_reward', 
                'mean_episode_return', 
                'std_episode_return', 
                'q25_episode_return', 
                'q50_episode_return',
                'q75_episode_return',
                'annualised_mean_episode_return',
                'annualised_std_episode_return', 
                'annualised_q25_episode_return', 
                'annualised_q50_episode_return', 
                'annualised_q75_episode_return', 
            ] + list(itertools.chain(*[(
                [
                    '{}_mean_episode_reward'.format(env_name),
                    '{}_std_episode_reward'.format(env_name),
                    '{}_q25_episode_reward'.format(env_name),
                    '{}_q50_episode_reward'.format(env_name),
                    '{}_q75_episode_reward'.format(env_name),
                    '{}_mean_episode_return'.format(env_name),
                    '{}_std_episode_return'.format(env_name),
                    '{}_q25_episode_return'.format(env_name),
                    '{}_q50_episode_return'.format(env_name),
                    '{}_q75_episode_return'.format(env_name),
                    '{}_annualised_mean_episode_return'.format(env_name),
                    '{}_annualised_std_episode_return'.format(env_name),
                    '{}_annualised_q25_episode_return'.format(env_name),
                    '{}_annualised_q50_episode_return'.format(env_name),
                    '{}_annualised_q75_episode_return'.format(env_name),
                ] + 
                ['{}_mean_{}'.format(env_name,e) for e in (['CASH'] + env.abbreviation)] + 
                ['{}_std_{}'.format(env_name,e) for e in (['CASH'] + env.abbreviation)] +
                ['{}_q25_{}'.format(env_name,e) for e in (['CASH'] + env.abbreviation)] +
                ['{}_q50_{}'.format(env_name,e) for e in (['CASH'] + env.abbreviation)] +
                ['{}_q75_{}'.format(env_name,e) for e in (['CASH'] + env.abbreviation)])
            for env_name,env in self.envs.items()])))

    def build_row(self, steps, policy):
        row = [str(steps)]
        rewards = {}
        actions = {}
        for env_name in self.envs:
            env = self.envs[env_name]
            num_episodes = self.num_episodes[env_name]
            rewards[env_name], actions[env_name] = rollout(env, policy, num_episodes)
        mean_reward, std_reward, q25_reward, q50_reward, q75_reward = \
            get_rewards_stats(list(itertools.chain(*rewards.values())))
        row.extend([
            '{:6f}'.format(mean_reward),
            '{:6f}'.format(std_reward),
            '{:6f}'.format(q25_reward),
            '{:6f}'.format(q50_reward),
            '{:6f}'.format(q75_reward)
        ])
        mean_return, std_return, q25_return, q50_return, q75_return = \
            get_rewards_stats(list(itertools.chain(*[[np.exp(v) for v in rewards] for rewards in rewards.values()])))
        row.extend([
            '{:6f}'.format(mean_return),
            '{:6f}'.format(std_return),
            '{:6f}'.format(q25_return),
            '{:6f}'.format(q50_return),
            '{:6f}'.format(q75_return)
        ])
        annualized_mean_return, annualized_std_return, annualized_q25_return, annualized_q50_return, annualized_q75_return = \
            get_rewards_stats(list(itertools.chain(*[[np.exp(v) ** (self.steps_per_year / self.envs[env_name].steps) for v in rewards[env_name]] for env_name in self.envs])))
        row.extend([
            '{:6f}'.format(annualized_mean_return),
            '{:6f}'.format(annualized_std_return),
            '{:6f}'.format(annualized_q25_return),
            '{:6f}'.format(annualized_q50_return),
            '{:6f}'.format(annualized_q75_return)
        ])
        for env_name in self.envs:
            mean_reward, std_reward, q25_reward, q50_reward, q75_reward = \
                get_rewards_stats(rewards[env_name])
            row.extend([
                '{:6f}'.format(mean_reward),
                '{:6f}'.format(std_reward),
                '{:6f}'.format(q25_reward),
                '{:6f}'.format(q50_reward),
                '{:6f}'.format(q75_reward)
            ])
            mean_return, std_return, q25_return, q50_return, q75_return = \
                get_rewards_stats([np.exp(v) for v in rewards[env_name]])
            row.extend([
                '{:6f}'.format(mean_return),
                '{:6f}'.format(std_return),
                '{:6f}'.format(q25_return),
                '{:6f}'.format(q50_return),
                '{:6f}'.format(q75_return)
            ])
            annualized_mean_return, annualized_std_return, annualized_q25_return, annualized_q50_return, annualized_q75_return = \
                get_rewards_stats([np.exp(v) ** (self.steps_per_year / self.envs[env_name].steps) for v in rewards[env_name]])
            row.extend([
                '{:6f}'.format(annualized_mean_return),
                '{:6f}'.format(annualized_std_return),
                '{:6f}'.format(annualized_q25_return),
                '{:6f}'.format(annualized_q50_return),
                '{:6f}'.format(annualized_q75_return)
            ])
            mean_action, std_action, q25_action, q50_action, q75_action = \
                get_actions_stats(actions[env_name])
            row.extend(['{:6f}'.format(e) for e in mean_action])
            row.extend(['{:6f}'.format(e) for e in std_action])
            row.extend(['{:6f}'.format(e) for e in q25_action])
            row.extend(['{:6f}'.format(e) for e in q50_action])
            row.extend(['{:6f}'.format(e) for e in q75_action])
        return row
