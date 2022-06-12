from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import collections
import numpy as np
import pickle
import random

from pathlib import Path

import ray
from ray.rllib.optimizers.replay_buffer import ReplayBuffer, \
    PrioritizedReplayBuffer, JiangReplayBuffer
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.compression import pack_if_needed
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.schedules import LinearSchedule
from ray.rllib.utils.memory import ray_get_and_free

from .custom_replay_buffer import PortfolioVectorMemory

logger = logging.getLogger(__name__)


class MultiUniversePortfolioOptimizer(PolicyOptimizer):
    """Variant of the local sync optimizer that supports replay (for DQN).

    This optimizer requires that rollout workers return an additional
    "td_error" array in the info return of compute_gradients(). This error
    term will be used for sample prioritization."""

    def __init__(self,
                 workers,
                 envs,
                 train_batch_size=32,
                 sampling_bias=1e-5,
                 dump_gradients=None):
        """Initialize an sync replay optimizer.

        Arguments:
            workers (WorkerSet): all workers
            learning_starts (int): wait until this many steps have been sampled
                before starting optimization.
            buffer_size (int): max size of the replay buffer
            prioritized_replay (bool): whether to enable prioritized replay
            prioritized_replay_alpha (float): replay alpha hyperparameter
            prioritized_replay_beta (float): replay beta hyperparameter
            prioritized_replay_eps (float): replay eps hyperparameter
            schedule_max_timesteps (int): number of timesteps in the schedule
            beta_annealing_fraction (float): fraction of schedule to anneal
                beta over
            final_prioritized_replay_beta (float): final value of beta
            train_batch_size (int): size of batches to learn on
            sample_batch_size (int): size of batches to sample from workers
            before_learn_on_batch (function): callback to run before passing
                the sampled batch to learn on
            synchronize_sampling (bool): whether to sample the experiences for
                all policies with the same indices (used in MADDPG).
        """
        PolicyOptimizer.__init__(self, workers)

        self.train_batch_size = train_batch_size
        self.training = False
        self.dump_gradients = dump_gradients

        # Stats
        self.update_weights_timer = TimerStat()
        self.sample_timer = TimerStat()
        self.replay_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.learner_stats = {}

        self.envs = envs
        self.pvms = {env_name: PortfolioVectorMemory(env.steps, bias=sampling_bias) 
                     for env_name, env in envs.items()}
        self.names = list(envs.keys())

    @override(PolicyOptimizer)
    def step(self):
        if not self.training:
            #batch = self.workers.local_worker().sample()
            #if isinstance(batch, SampleBatch):
            #    batch = MultiAgentBatch({
            #        DEFAULT_POLICY_ID: batch
            #    }, batch.count)
            #policy = self.workers.local_worker().policy_map[DEFAULT_POLICY_ID]
            #for row in batch.policy_batches[DEFAULT_POLICY_ID].rows():
            #    self.replay_buffers[DEFAULT_POLICY_ID].add(
            #        pack_if_needed(row["obs"]),
            #        row["actions"],
            #        row["rewards"],
            #        pack_if_needed(row["new_obs"]),
            #        row["dones"])
            for env_name,env in self.envs.items():
                done = False
                obs = env.reset()
                act = np.ones(env.action_space.shape[0]) / env.action_space.shape[0]
                while not done:
                    next_obs, rew, done, info = env.step(act)
                    self.pvms[env_name].add(obs, act, rew, next_obs, done)
                    obs = next_obs
            self.training = True
        self._optimize()

    @override(PolicyOptimizer)
    def stats(self):
        return dict(
            PolicyOptimizer.stats(self), **{
                "sample_time_ms": round(1000 * self.sample_timer.mean, 3),
                "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
                "grad_time_ms": round(1000 * self.grad_timer.mean, 3),
                "update_time_ms": round(1000 * self.update_weights_timer.mean,
                                        3),
                "opt_peak_throughput": round(self.grad_timer.mean_throughput,
                                             3),
                "opt_samples": round(self.grad_timer.mean_units_processed, 3),
                "learner": self.learner_stats,
            })

    def _optimize(self):

        if self.dump_gradients:
            dump_data = {}

        name = random.choice(self.names)
        pvm = self.pvms[name]

        samples = {}
        idxes = None
        idxes = pvm.sample_idxes(self.train_batch_size)
        (obses_t, actions, rewards, obses_tp1, dones) = pvm.sample_with_idxes(idxes)
        weights = np.ones_like(rewards)
        batch_indexes = -np.ones_like(rewards)
        samples[DEFAULT_POLICY_ID] = SampleBatch({
            "obs": obses_t,
            "actions": actions,
            "rewards": rewards,
            "new_obs": obses_tp1,
            "dones": dones,
            "weights": weights,
            "batch_indexes": batch_indexes
        })
        samples = MultiAgentBatch(samples, self.train_batch_size)

        # dump idxes
        # dump (obses_t, actions, rewards, obses_tp1, dones)
        # dump gradients
        # dump policy weights
        # dump idxes
        # dump (obses_t, actions, rewards, obses_tp1, dones)
        # dump next_obs_batch

        info = self.workers.local_worker().learn_on_batch(samples)[DEFAULT_POLICY_ID]

        if self.dump_gradients:
            dump_data['idxes_t'] = idxes
            dump_data['batch_t'] = (obses_t, actions, rewards, obses_tp1, dones)
            dump_data['grads_and_vars'] = info['grads_and_vars']
            dump_data['next_obs'] = info['next_obs_batch']

        idxes = [idx+1 for idx in idxes]
        if idxes[-1] > pvm._maxsize - 1:
            idxes = idxes[:-1]
        next_obs_batch = info['next_obs_batch']
        (obses_t, actions, rewards, obses_tp1, dones) = pvm.sample_with_idxes(idxes)
        for obs_t, action, reward, obs_tp1, done, idx in zip(next_obs_batch, actions, rewards, obses_tp1, dones, idxes):
            pvm.modify(obs_t, action, reward, obs_tp1, done, idx)

        if self.dump_gradients:
            dump_data['idxes_t2'] = idxes
            dump_data['batch_t2'] = (obses_t, actions, rewards, obses_tp1, dones)

        self.num_steps_sampled += 1

        if self.dump_gradients:
            pickle.dump(dump_data, (Path(self.dump_gradients) / 'data_{}.pkl'.format(str(self.num_steps_sampled))).open('wb'))


def make_policy_optimizer(workers, envs, config):
    return MultiUniversePortfolioOptimizer(
        workers,
        envs,
        train_batch_size=config["train_batch_size"],
        sampling_bias=config["replay_bias"],
        dump_gradients=config["dump_gradients"],
        **config["optimizer"])
