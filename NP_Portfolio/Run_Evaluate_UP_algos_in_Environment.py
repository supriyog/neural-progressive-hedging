import collections
import importlib
import json
import operator
import os
import pickle
import shelve
import sys
import shutil
import copy
from pathlib import Path

import argparse
import gym
import numpy as np
import yaml

from utils import utilities

from pathlib import Path
from data import stock_loading
from data import data_loading

cur_work_dir = os.getcwd()+"/"

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument("--cf", type=str, default="config/config_snp500_np.txt")
parser.add_argument("--rf", type=str, default="result_NPC_v1")
parser.add_argument("--ef", type=str, default="exp01")
parser.add_argument("--sn", type=str, default="0")
parser.add_argument("--en", type=str, default="0")
parser.add_argument("--an", type=str, default="0")
parser.add_argument("--uf", type=str, default="UPweights01")

args = parser.parse_args()
config_file = cur_work_dir+args.cf
result_file = args.rf
experiment_file = args.ef
experiment_number = int(args.en)
algo_name_ID = int(args.an)
stock_selction_number = int(args.sn)
upweights_dir = args.uf

mod = sys.modules[__name__]


import pandas as pd
all_stocks_data = pd.read_csv("data/stock_universes.csv", header=None)
all_ten_stocks = [[all_stocks_data.iloc[i][j] for j in range(len(all_stocks_data.iloc[i]))] for i in range(len(all_stocks_data))]
stocks = all_ten_stocks[stock_selction_number]
if stock_selction_number >= 4:
	print ("Experimental results in the paper are presented with first four universe -- the selected univese number is: "+str(stock_selction_number))

# ========== 3. data configuration ========== #

import environment.sp500_data_loader as sp500_data_loader
import environment.k_markov_data_loader as k_markov_data_loader
import environment.gbm_data_loader as gbm_data_loader


def get_train_data_config():
    start_date = '2009-07-01'
    end_date = '2019-01-01'
    data_loc = 'data/sandp500_2005_2019/{}_data.csv'
    return {'start_date': start_date, 'end_date': end_date, 'data_loc': data_loc}

def get_validation_data_config():
    start_date = '2005-01-01'
    end_date = '2017-01-01'
    data_loc = 'data/sandp500_2005_2019/{}_data.csv'
    idx_of_date = '2016-01-01'
    return {'start_date': start_date, 'end_date': end_date,
            'data_loc': data_loc, 'idx_of_date': idx_of_date}

def get_test_data_config():
    start_date = '2009-01-01'
    end_date = '2020-01-01'
    data_loc = 'data/sandp500_2005_2019/{}_data.csv'
    idx_of_date = '2019-01-01'
    return {'start_date': start_date, 'end_date': end_date,
            'data_loc': data_loc, 'idx_of_date': idx_of_date}


train_data_config = get_train_data_config()

train_data_config['stocks'] = stocks
train_data_loader = 'sp500_data_loader.get_data'
train_history, start_idx = operator.attrgetter(train_data_loader)(mod)(**train_data_config)
train_data = (train_history, stocks)


test_data_config = get_test_data_config()
test_data_config['stocks'] = stocks
test_data_loader = 'sp500_data_loader.get_data'
test_history, start_idx = operator.attrgetter(test_data_loader)(mod)(**test_data_config)
test_data = (test_history, stocks)

train_price_vector = train_history
eval_price_vector = test_history[:,start_idx:,:]


#=========================== SP config starts here =====================#
cfile = open(config_file)
line = cfile.readline()
param_dict = {}
while line != "":
	if "	#" in line:
		line = line.strip(" \r\n")
		line = line.split("	#")
		param_dict[line[1].split(" ")[0]] = line[0]
	line = cfile.readline()
cfile.close()

num_assets = len(stocks)    #int(param_dict["num_asset"])
num_liability = int(param_dict["num_liability"])
horizon_length = int(param_dict["horizon_length"])
look_ahead_period = int(param_dict["look_ahead_period"])
init_cash = float(param_dict["init_cash"])
start_evaluation_day = int(param_dict["start_eval_day"])
start_writing_result = str(param_dict["starting_result"])
prev_holding_allocation = [0.0 for i in range(num_assets + 1)]
alpha_plus = float(param_dict["alpha_plus"])  # 0.01	#alpha_plus
alpha_minus = float(param_dict["alpha_minus"])  # 0.01	#alpha_minus
n_sample_scenario = [0 for lt in range(look_ahead_period - 1)]
for lt in range(1, look_ahead_period):
	n_sample_scenario[lt - 1] = int(param_dict["n_sample_scenario_" + str(lt)])


interest_rates_eval = np.zeros((len(stocks), eval_price_vector.shape[1] - 1))
for time_step in range(eval_price_vector.shape[1] - 1):
	interest_rates_eval[:, time_step] = 1.0 * (eval_price_vector[:, time_step + 1, 3] - eval_price_vector[:, time_step, 3]) / (1.0 * eval_price_vector[:, time_step, 3])

#=========================== SP config ends here =====================#

max_eval_steps = 30
algo_names = ["OLMAR", "PAMR", "RMR", "UCRP"]

all_rewards = np.array([])
all_weights = np.array([])

UP_actions = np.array([[1.0/(len(stocks)+1) for i in range(len(stocks)+1)] for j in range(max_eval_steps)])
if(algo_name_ID < 3):
	UP_actions = np.load(cur_work_dir+'experiments/{}/UP_weights_{}_{}_{}.npy'.format(upweights_dir,algo_names[algo_name_ID],stock_selction_number,experiment_number))

exp_name = "exp4004"#+str(stock_selction_number+1)
step_num = "1500000"
stock_desc = "pq_testing"
data_desc = "random_"+str(stock_selction_number)
trading_cost = 0.002

exp_dirpath = Path('experiments') / exp_name
base_env_config = yaml.load((exp_dirpath / 'base_env_config.yaml').open('r'))
window_length = base_env_config['base_env_params']['window_length']
env_config = {
	'base_env_params': {
		'steps': 0,
		'window_length': 3,
		'trading_cost': trading_cost,
	}
}

env_config['base_env_params']['window_length'] = window_length

import environment.portfolio_env as pf


for exp_num in range(experiment_number,experiment_number+1):
	test_data = (test_history[:,max_eval_steps*exp_num:,:], stocks)

	current_weights = np.array([])
	# ========== 4. environment configuration ========== #

	kwargs = copy.deepcopy(env_config['base_env_params'])
	kwargs['start_idx'] = start_idx

	test_env = pf.PortfolioEnv(data = test_data, **kwargs)

	time_step = 0
	init_allocation = [0.0 for asst in range(num_assets)]
	init_cash = 1.0
	init_allocation.append(init_cash)

	rewards = [0]
	obs = np.expand_dims(test_env.reset(), 0)
	done = False
	eval_steps = 0
	all_interest_samples = np.array([])
	while not done:
		action_to_ddpg = UP_actions[time_step]
		current_weights = action_to_ddpg if len(current_weights) == 0 else np.vstack((current_weights, action_to_ddpg))

		#=========================== SP logic ends here =====================#
		next_obs, reward, done, info = test_env.step(action_to_ddpg)
		init_allocation_ddpg_order = info['final_portfolio_value'] * info['final_portfolio_weights']
		init_allocation_ddpg = init_allocation_ddpg_order[1:]
		init_allocation_ddpg = np.append(init_allocation_ddpg, init_allocation_ddpg_order[0])
		obs = np.expand_dims(next_obs, 0)
		rewards.append(reward)
		eval_steps += 1
		time_step += 1
		if eval_steps >= max_eval_steps:
			done = True
			returns = np.exp(np.cumsum(rewards))
			all_rewards = [returns] if len(all_rewards) == 0 else np.vstack((all_rewards, returns))
			all_weights = np.array([current_weights]) if len(all_weights) == 0 else np.vstack((all_weights, [current_weights]))


np.save(cur_work_dir+'results/{}/Returns_{}_{}_{}_{}.npy'.format(experiment_file,result_file,algo_names[algo_name_ID],stock_selction_number,experiment_number), all_rewards)
np.save(cur_work_dir + 'results/{}/Weights_{}_{}_{}_{}.npy'.format(experiment_file,result_file,algo_names[algo_name_ID],stock_selction_number,experiment_number), all_weights)
