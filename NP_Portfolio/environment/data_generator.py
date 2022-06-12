import numpy as np
from utils.data import date_to_index, index_to_date

class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(self, history, steps=730, window_length=50, start_idx=0): #, start_idx=0, start_date=None):
        """

        Args:
            history: (num_stocks, timestamp, 5) open, high, low, close, volume
            abbreviation: a list of length num_stocks with assets name
            steps: the total number of steps to simulate, default is 2 years
            window_length: observation window, must be less than 50
            start_idx: if >0, this is the first index
            start_date: the date to start. Default is None and random pick one.
                        It should be a string e.g. '2012-08-13'
        """
        #assert history.shape[0] == len(abbreviation), 'Number of stock is not consistent'
        #import copy

        self.steps = steps  # + 1
        self.window_length = window_length
        #self.start_idx = start_idx
        #self.start_date = start_date

        # make immutable class
        self._data = history.copy()  # all data
        #self.asset_names = copy.copy(abbreviation)
        self.start_idx = start_idx

    def _step(self):
        # get observation matrix from history, exclude volume, maybe volume is useful as it
        # indicates how market total investment changes. Normalize could be critical here
        self.step += 1
        obs = self.data[:, self.step:self.step + self.window_length, :].copy()
        # normalize obs with open price

        # used for compute optimal action and sanity check
        ground_truth_obs = self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()

        done = self.step >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        self.step = 0

        # get data for this episode, each episode might be different.
        #if self.start_date is None:
        if self.steps == 0:
            self.idx = self.window_length + self.start_idx
        else:
            self.idx = np.random.randint(
                low=self.window_length + self.start_idx, high=self._data.shape[1] - self.steps)
        #else:
        #    # compute index corresponding to start_date for repeatable sequence
        #    self.idx = date_to_index(self.start_date) - self.start_idx
        #    assert self.idx >= self.window_length and self.idx <= self._data.shape[1] - self.steps, \
        #        'Invalid start date, must be window_length day after start date and simulation steps day before end date'
        # print('Start date: {}'.format(index_to_date(self.idx)))
        data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :5]
        # apply augmentation?
        self.data = data
        return self.data[:, self.step:self.step + self.window_length, :].copy(), \
               self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()
