from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import sys

from ray.rllib.optimizers.segment_tree import SumSegmentTree, MinSegmentTree
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.rllib.utils.window_stat import WindowStat


@DeveloperAPI
class PortfolioVectorMemory(object):
    @DeveloperAPI
    def __init__(self, size, bias):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._hit_count = np.zeros(size)
        self._eviction_started = False
        self._num_added = 0
        self._num_sampled = 0
        self._evicted_hit_stats = WindowStat("evicted_hit", 1000)
        self._est_size_bytes = 0
        self._bias = bias
        weights = []
        p_min = bias * ((1 - bias) ** (size-1))
        max_weight = (p_min * size) ** (-1)
        for idx in range(size):
            relative_idx = idx % size
            p_sample = bias * ((1 - bias) ** (size-relative_idx-1))
            weight = (p_sample * size) ** (-1)
            weights.append(weight/max_weight)
        self._weights = weights

    def __len__(self):
        return len(self._storage)

    @DeveloperAPI
    def add(self, obs_t, action, reward, obs_tp1, done):
        if len(self._storage) == self._maxsize:
            raise
        data = (obs_t, action, reward, obs_tp1, done)
        self._num_added += 1
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            self._est_size_bytes += sum(sys.getsizeof(d) for d in data)
        else:
            self._storage[self._next_idx] = data
        if self._next_idx + 1 >= self._maxsize:
            self._eviction_started = True
        self._next_idx = (self._next_idx + 1) % self._maxsize
        if self._eviction_started:
            self._evicted_hit_stats.push(self._hit_count[self._next_idx])
            self._hit_count[self._next_idx] = 0

    def modify(self, obs_t, action, reward, obs_tp1, done, idx):
        if idx < len(self._storage):
            self._storage[idx] = (obs_t, action, reward, obs_tp1, done)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(unpack_if_needed(obs_t), copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(unpack_if_needed(obs_tp1), copy=False))
            dones.append(done)
            self._hit_count[i] += 1
        return (np.array(obses_t), np.array(actions), np.array(rewards),
                np.array(obses_tp1), np.array(dones))

    @DeveloperAPI
    def sample_idxes(self, batch_size):
        n = self._maxsize
        bias = self._bias

        idx = np.random.geometric(bias) - 1
        while idx > n - batch_size:
            idx = np.random.geometric(bias) - 1
        idx = n - batch_size - idx
        idxes = list(range(idx,idx+batch_size))

        return idxes

    @DeveloperAPI
    def sample_with_idxes(self, idxes):
        self._num_sampled += len(idxes)
        return self._encode_sample(idxes)

    @DeveloperAPI
    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
          batch of observations
        act_batch: np.array
          batch of actions executed given obs_batch
        rew_batch: np.array
          rewards received as results of executing act_batch
        next_obs_batch: np.array
          next set of observations seen after executing act_batch
        done_mask: np.array
          done_mask[i] = 1 if executing act_batch[i] resulted in
          the end of an episode and 0 otherwise.
        """

        n = self._maxsize
        bias = self._bias

        idx = np.random.geometric(bias) - 1
        while idx > n - batch_size:
            idx = np.random.geometric(bias) - 1
        idx = n - batch_size - idx
        idxes = list(range(idx,idx+batch_size))

        self._num_sampled += batch_size

        return self._encode_sample(idxes)

    @DeveloperAPI
    def stats(self, debug=False):
        data = {
            "added_count": self._num_added,
            "sampled_count": self._num_sampled,
            "est_size_bytes": self._est_size_bytes,
            "num_entries": len(self._storage),
        }
        if debug:
            data.update(self._evicted_hit_stats.stats())
        return data
