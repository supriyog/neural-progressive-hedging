"""
Modified from https://github.com/wassname/rl-portfolio-management/blob/master/src/environments/portfolio.py
"""
from __future__ import print_function

from pprint import pprint

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import gym
import gym.spaces



from utils.data import date_to_index, index_to_date
from environment_constaint.data_generator import DataGenerator
from environment_constaint.augmented_data_generator import DataGenerator as AugmentedDataGenerator
from environment_constaint.portfolio_sim import PortfolioSim

eps = 1e-8

class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 steps=730,  # 2 years
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 #sample_start_date=None,
                 data=None,
                 start_idx=None,
                 init_weights='cash',
                 augment_synthetic_data=False,
                 synthetic_data_config=None,
                 cash_req=None,
                 seed = 10001
                 ):
        """
        An environment for financial portfolio management.
        Params:
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
            sample_start_date - The start date sampling from the history
            start_idx - if None, no effect, if >0, this is the row 
                corresponding to the price change of the first valid trading day, and therefore,
                if >0, this must be >= window_length, and in other words, 
                choosing start_idx = window_length is equivalent to choosing start_idx = None
        """
        self._seed = seed
        if not augment_synthetic_data:
            history, self.abbreviation = data
        else:
            history, dates, self.abbreviation = data
            synthetic_data_config = synthetic_data_config or dict()

        #history, self.abbreviation = self.get_data(start_date, end_date, stocks)
        #history = all_hist[:, 900:, :]
        #tr_hist = all_hist[:, :900, :]
        if start_idx is None:
            start_idx = window_length
        self.steps_test = steps
        if steps == 0:
            steps = history.shape[2] - start_idx - 1

        self.steps = steps
        #print ("number of steps: "+ str(history.shape[1]) +" "+ str(start_idx) +" "+ str(self.steps))

        self.window_length = window_length
        #print ("window length is: "+str(self.window_length))
        self.num_stocks = history.shape[1]
        self.init_weights = init_weights
        self.cash_requirments = np.zeros(self.steps)
        if(self.steps_test == 0):
            self.cash_requirments = cash_req
        self.step_value = 0
        if self.init_weights == 'cash':
            self.init_action = np.zeros(self.num_stocks+1)
            self.init_action[0] = 1.
        elif self.init_weights == 'eq':
            self.init_action = np.ones(self.num_stocks+1) / (self.num_stocks+1)
        else:
            raise

        if not augment_synthetic_data:
            self.src = DataGenerator(history, steps=steps, window_length=window_length, start_idx=start_idx-window_length, steps_test= self.steps_test)
        else:
            self.src = AugmentedDataGenerator(history, dates, self.abbreviation, steps=steps, window_length=window_length, start_idx=start_idx-window_length, synthetic_data_config=synthetic_data_config)

        self.sim = PortfolioSim(
            num_stocks=self.num_stocks,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps, init_weights=init_weights)

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(self.num_stocks + 1,), dtype=np.float32)  # include cash

        # get the observation space from the data min and max
        #self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.abbreviation) + 1, window_length,
        #                                                                         history.shape[-1] - 1), dtype=np.float32)
        # desmond added
        #self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stocks + 1, window_length + 1, history.shape[-1]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(((self.num_stocks + 1) * (window_length + 1) * history.shape[-1])+2, ), dtype=np.float32)
        #print ("state and action space: "+str(self.observation_space)+ " "+ str(self.action_space))
    def step(self, action):
        return self._step(action)

    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        if(len(action.shape) > 1):
            action = action.reshape(action.shape[1])

        action_convert = np.exp(action - np.max(action))
        action = action_convert/np.sum(action_convert)  #using softmax to normalize weight actions
        #print ("action value from step is: "+str(action))
        np.testing.assert_almost_equal(
            action.shape,
            (self.num_stocks + 1,)
        )

        # normalise just in case
        action = np.clip(action, 0, 1)

        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights /= (weights.sum() ) #+ eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)  # so if weights are all zeros we normalise to [1,0...]
        print("action is: " + str(action))

        #assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        #np.testing.assert_almost_equal(
        #    np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        observation, done1, ground_truth_obs = self.src._step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # desmond added
        # need to include previous action in the state (cf. Jiang)
        # note that original dimensions of observation are 
        #   [num_assets,window_length,num_features]
        # note that the dimension of the previous action is [num_assets]
        # we append the actions to the window_length axis so that the new
        # dimensions of the observation are now
        #   [num_assets, window_length+1, num_features]
        # when num_features > 1, we tile the previous actions accordingly
        action_expanded = np.expand_dims(np.tile(action.reshape((-1,1)), (1,5)), axis=1)
        observation = np.concatenate((observation, action_expanded), axis=1)

        # relative price vector of last observation day (close/open)
        #close_price_vector = observation[:, -2, 3] #-1
        close_price_vector = observation[:, -3, 3] #-1
        #open_price_vector = observation[:, -1, 0]
        #open_price_vector = observation[:, -3, 3] #changed to using closing price only
        open_price_vector = observation[:, -4, 3] #changed to using closing price only
        y1 = close_price_vector / open_price_vector

        #print ("weights executed in environment is: "+str(weights))
        reward, info, done2 = self.sim._step(weights, y1) #old reward, not needed

        #p1_prime= self.sim.p0 * np.dot(observation[:, -1, 3]/observation[:, -2, 3], weights)
        p1_prime= self.sim.p0 * np.dot(observation[:, -2, 3]/observation[:, -3, 3], weights)
        reward=np.log(p1_prime/self.sim.p1_prime) #/self.sim.steps*1000 


        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        #info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step)
        info['steps'] = self.src.step
        #info['next_obs'] = ground_truth_obs
        #info['portfolio_value']=   p1_prime
        #info['final_portfolio_value']=   self.sim.p0 * np.dot(observation[:, -1, 3]/observation[:, -2, 3],
        #                                                      self.sim.weights) # final portfolio value (after last step)
        info['final_portfolio_value']=   self.sim.p0 * np.dot(observation[:, -2, 3]/observation[:, -3, 3],
                                                              self.sim.weights) # final portfolio value (after last step)
        #info['final_reward']=np.log(info['final_portfolio_value']/self.sim.p0)/self.sim.steps*1000
        info['final_reward']=np.log(info['final_portfolio_value']/self.sim.p0)
        weights = (observation[:, -2, 3]/observation[:, -3, 3]) * self.sim.weights # final portfolio weights (after last step)
        info['final_portfolio_weights'] = weights / np.sum(weights)

        cost_requirements = self.cash_requirments[self.step_value]   #change this value to cash requirements according to this time step.
        info['cost'] = 0.0
        print ("cost demand supply is at "+str(self.step_value)+" : "+str(self.cash_requirments[self.step_value])+"  "+str(info['final_portfolio_value']*info['final_portfolio_weights'][0]))
        if(info['final_portfolio_value']*info['final_portfolio_weights'][0] < cost_requirements):
            info['cost'] = 1.0
        self.infos.append(info)
        #print (np.array(observation).shape)
        observation = np.array(observation).flatten()
        observation = np.append(observation, np.array([self.step_value+1, self.cash_requirments[self.step_value]]))
        self.step_value += 1
        # append time step and previous cash requirements information into observation array.

        return observation, reward, done1 or done2, info
    
    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ground_truth_obs, cash_requirment = self.src.reset()
        if self.steps_test >= 1:
            #print ("enter bad testing settings")
            self.cash_requirments = cash_requirment
        self.step_value = 0
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        #print (cash_observation.shape, observation.shape)
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # desmond added
        #action_expanded = np.expand_dims(np.tile(np.ones((self.num_stocks+1,1))/(self.num_stocks+1), (1,5)), axis=1)
        action_expanded = np.expand_dims(np.tile(np.expand_dims(self.init_action,axis=1), (1,5)), axis=1)
        observation = np.concatenate((observation, action_expanded), axis=1)
        #print ("observation from reset shape is: "+str(observation.shape))

        info = {}
        #info['next_obs'] = ground_truth_obs
        observation = np.array(observation).flatten()
        observation = np.append(observation, np.array([0.0, 0.0]))
        #observation = np.append(observation, np.array([0,0])) #add this line after adding cost constraints.
        return observation

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            #pprint(self.infos[-1])
            info = self.infos[-1]
            df_info = pd.DataFrame(self.infos)
            if(info["steps"] == self.steps):
                print('Final Portfolio Value : ',info["final_portfolio_value"] )
                print('Sharpe Ratio ', self.sharpe(df_info.rate_of_return))


        elif mode == 'human':
            self.plot()
            
    def render(self, mode='human', close=False):
        return self._render(mode='ansi', close=False)

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        print(df_info)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = self.max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = self.sharpe(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf(), rot=30)

    def random_shift(x, fraction):
        """ Apply a random shift to a pandas series. """
        min_x, max_x = np.min(x), np.max(x)
        m = np.random.uniform(-fraction, fraction, size=x.shape) + 1
        return np.clip(x * m, min_x, max_x)

    def scale_to_start(x):
        """ Scale pandas series so that it starts at one. """
        x = (x + eps) / (x[0] + eps)
        return x

    @staticmethod
    def sharpe(returns, freq=30, rfr=0):
        """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
        return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)

    @staticmethod
    def max_drawdown(returns):
        """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
        peak = returns.max()
        trough = returns[returns.argmax():].min()
        return (trough - peak) / (peak + eps)

