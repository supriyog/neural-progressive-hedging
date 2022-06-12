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

import ray
from ray.rllib.utils import try_import_tf
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from OptimizerCodes.progressiveHedgingPO_dl_CVAR import stocProgPH
from utils import utilities

cur_work_dir = os.getcwd()+"/"

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument("--cf", type=str, default="../config/config_snp500_np.txt")
parser.add_argument("--rf", type=str, default="result_NPC_v1")
parser.add_argument("--ef", type=str, default="exp01")
parser.add_argument("--en", type=str, default="0")
parser.add_argument("--sn", type=str, default="0")
parser.add_argument("--cv", type=str, default="0.99")
parser.add_argument("--rld", type=str, default="exp4004")
parser.add_argument("--cp", type=str, default="1500000")

args = parser.parse_args()
config_file = cur_work_dir+args.cf
result_file = args.rf
experiment_file = args.ef
experiment_number = int(args.en)
stock_selction_number = int(args.sn)
alpha_cvar = float(args.cv)
checkpoint_number = str(args.cp)
rl_dir = str(args.rld)

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
train_price_vector = train_history
test_data = (test_history, stocks)


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

train_interest_rates_all = np.zeros((len(stocks), train_price_vector.shape[1] - 1))
for time_step in range(train_price_vector.shape[1] - 1):
	train_interest_rates_all[:, time_step] = np.log(1.0 * (train_price_vector[:, time_step + 1, 3]) / (1.0 * train_price_vector[:, time_step, 3]))
train_interests_mean = np.mean(train_interest_rates_all, axis = 1)
train_interests_var = np.std(train_interest_rates_all, axis = 1)
train_interests_cov = np.cov(train_interest_rates_all)


#=========================== SP config ends here =====================#

json_write_file = cur_work_dir + "data/json_samples/scenario_" + str(result_file) + ".json"
if(not os.path.exists(cur_work_dir + "results/"+experiment_file+"/")):
	os.makedirs(cur_work_dir + "results/"+experiment_file+"/")

if(os.path.exists(cur_work_dir + "results/" + experiment_file + "/" + str(result_file) + "_" +str(stock_selction_number) +"_" + str(alpha_cvar) +"_" + str(experiment_number) + ".txt")):
	shutil.move(cur_work_dir + "results/" + experiment_file + "/" + str(result_file) + "_" +str(stock_selction_number) +"_" + str(alpha_cvar) +"_" + str(experiment_number) + "_backup.txt")
sp_result_file = cur_work_dir + "results/" + experiment_file + "/" + str(result_file) + "_" +str(stock_selction_number) +"_" + str(alpha_cvar) +"_" + str(experiment_number) + ".txt"

max_eval_steps = 30
max_exp_numbers = 5

all_weights = np.array([])
all_rewards = np.array([])


exp_name = rl_dir
step_num = checkpoint_number #"1500000"
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

tf = try_import_tf()
tf.reset_default_graph()
tf.Session().__enter__()

kwargs = copy.deepcopy(env_config['base_env_params'])
kwargs['start_idx'] = start_idx
test_env = pf.PortfolioEnv(data = test_data, **kwargs)

with tf.variable_scope(DEFAULT_POLICY_ID):
	policy_module = 'experiments.{}.custom_policy'.format(exp_name)
	rllib_trainer_config = yaml.load((exp_dirpath / 'rllib_trainer_config.yaml').open('r'))
	obs_space = test_env.observation_space
	act_space = test_env.action_space
	policy = importlib.import_module(policy_module).CustomPolicy(obs_space, act_space, rllib_trainer_config)
with policy.sess.graph.as_default():
	policy_checkpoint = str(exp_dirpath / 'policies' / step_num)
	tf.train.Saver().restore(policy.sess, policy_checkpoint)

for exp_num in range(experiment_number,experiment_number+1):
	test_data = (test_history[:,max_eval_steps*exp_num:,:], stocks)

	eval_price_vector = test_history[:,max_eval_steps*exp_num:,:]

	# ========== 4. environment configuration ========== #

	kwargs = copy.deepcopy(env_config['base_env_params'])
	kwargs['start_idx'] = start_idx
	test_env = pf.PortfolioEnv(data = test_data, **kwargs)

	interest_rates_eval = np.zeros((len(stocks), eval_price_vector.shape[1]-1))
	for time_step in range(eval_price_vector.shape[1]-1):
		interest_rates_eval[:, time_step] = 1.0 * (eval_price_vector[:, time_step + 1, 3] - eval_price_vector[:, time_step, 3]) / (1.0 * eval_price_vector[:, time_step, 3])

	time_step = 0
	init_allocation = [0.0 for asst in range(num_assets)]
	init_cash = 1.0
	init_allocation.append(init_cash)
	current_weights = np.array([])

	rewards = [0]
	obs = np.expand_dims(test_env.reset(), 0)
	done = False
	eval_steps = 0
	all_interest_samples = np.array([])
	while not done:
		action = policy.evaluate(obs).flatten()

		#================ SP logic will go here. send the action to SP and update the actions according to SP suggestion ================#
		interest_samples = {}
		for ts in range(look_ahead_period):
			interest_samples[ts] = {}
		n_cur_scen = 1
		cash_return = 0

		for ts in range(look_ahead_period - 1):
			for ns in range(n_cur_scen):
				ln_samples = np.random.multivariate_normal(train_interests_mean, train_interests_cov, size = n_sample_scenario[ts])
				samples = np.exp(ln_samples) - 1
				interest_samples[ts][ns] = samples
				if(len(all_interest_samples)==0):
					all_interest_samples = samples
				else:
					all_interest_samples = np.vstack([all_interest_samples, samples])
			n_cur_scen *= n_sample_scenario[ts]

		if (os.path.exists(cur_work_dir + 'data/prestored_samples/all_interests_samples_{}.npy'.format(stock_selction_number))):
			if ((look_ahead_period == 2) and (n_sample_scenario[0] == 1000)):
				interest_samples[0][0] = np.load(cur_work_dir + 'data/prestored_samples/all_interests_samples_{}.npy'.format(stock_selction_number))\
					[max_eval_steps*exp_num*n_sample_scenario[0]+time_step*n_sample_scenario[0]:max_eval_steps*exp_num*n_sample_scenario[0]+(time_step+1)*n_sample_scenario[0],:]

		tree_location = utilities.generate_scenario_tree(look_ahead_period, n_sample_scenario, interest_samples,cash_return, json_write_file)

		probInstance = stocProgPH(tree_location, num_assets, config_file, init_allocation, sp_result_file, alpha_cvar)  # pass horizon_length, num_assets, initial_values (x[0]), selling transc alpha_plus, buying transc alpha_minus
		action_to_sp = action[1:]
		action_to_sp = np.append(action_to_sp, action[0])
		root_node_results, root_transaction = probInstance.solveStocProgMaster(action_to_sp)

		#=========================== init_allocation should come from next_obs of environment from previous iteration .. for now it is taken from SP simulation ====#
		for asst in range(num_assets + 1):
			prev_holding_allocation[asst] = init_allocation[asst]

		for asst in range(num_assets):
			root_transaction[asst][0] = max(0, (1.0 * (root_node_results[asst] - init_allocation[asst]) / (1 - alpha_plus)))
			root_transaction[asst][1] = max(0, (1.0 * (init_allocation[asst] - root_node_results[asst]) / (1 + alpha_minus)))

		init_allocation = [0.0 for i in range(num_assets + 1)]
		current_allocation = np.array([0.0 for i in range(num_assets + 1)])
		total_cash_change = 0.0
		#=========================== Check if the asset can be sold after considering transaction cost ===========#
		for asst in range(num_assets):
			if ((1 + alpha_minus) * root_transaction[asst][1] > prev_holding_allocation[asst]):
				root_transaction[asst][1] = (1.0 * prev_holding_allocation[asst]) / (1 + alpha_minus)

		#=========================== Check if all assets can be bought (cash + sold >= bought); otherwise reweight proportionally ======#
		if (prev_holding_allocation[num_assets] + sum(root_transaction[i][1] for i in range(num_assets)) < sum(root_transaction[j][0] for j in range(num_assets))):
			total_avail_cash = prev_holding_allocation[num_assets] + sum(root_transaction[i][1] for i in range(num_assets))
			total_expected_buy = sum(root_transaction[j][0] for j in range(num_assets))
			conversion_ratio = (1.0 * total_avail_cash) / total_expected_buy
			for asst in range(num_assets):
				root_transaction[asst][0] = root_transaction[asst][0] * conversion_ratio

		for asst in range(num_assets):
			current_holding = prev_holding_allocation[asst] + (1 - alpha_plus) * root_transaction[asst][0] - (1 + alpha_minus) * root_transaction[asst][1]
			total_cash_change += (root_transaction[asst][1] - root_transaction[asst][0])
			init_allocation[asst] = (1 + interest_rates_eval[asst][time_step + 1]) * current_holding
			current_allocation[asst] = current_holding
		print("SP interest rate: " + str(1 + interest_rates_eval[:, time_step + 1]))
		final_policy_value = [root_transaction[asst][0] - root_transaction[asst][1] for asst in range(num_assets)]

		init_allocation[num_assets] = (1 + cash_return) * (prev_holding_allocation[num_assets] + total_cash_change)
		current_allocation[num_assets] = prev_holding_allocation[num_assets] + total_cash_change
		utilities.write_results("SP average alloc results in exp " + str(exp_num) + " iteration "+str(eval_steps)+" is ## " + str(sum(root_node_results)) + " ## " + str(root_node_results), sp_result_file)
		utilities.write_results("Current holding value in exp " + str(exp_num) + " iteration "+str(eval_steps)+" ## " + str(sum(current_allocation)) + " ## " + str(current_allocation), sp_result_file)
		utilities.write_results("Buy-sell policy is in exp " + str(exp_num) + " iteration "+str(eval_steps)+" ## " + str(sum(final_policy_value)) + " ## " + str(final_policy_value),sp_result_file)
		utilities.write_results("SP_Portfolio value in exp " + str(exp_num) + " next period in iteration "+str(eval_steps)+" ## " + str(sum(init_allocation)) + " ## " + str(init_allocation),sp_result_file)

		#=========================== convert SP policy to weights and update action for RL evaluation ========#
		action_to_ddpg_sp_order = current_allocation / np.sum(current_allocation)
		action_to_ddpg = action_to_ddpg_sp_order[-1]
		action_to_ddpg = np.append(action_to_ddpg, action_to_ddpg_sp_order[0:-1])
		print("action to ddpg: " + str(action_to_ddpg))
		#=========================== SP logic ends here =====================#
		current_weights = action_to_ddpg if len(current_weights) == 0 else np.vstack((current_weights, action_to_ddpg))

		next_obs, reward, done, info = test_env.step(action_to_ddpg)
		init_allocation_ddpg_order = info['final_portfolio_value'] * info['final_portfolio_weights']
		init_allocation_ddpg = init_allocation_ddpg_order[1:]
		init_allocation_ddpg = np.append(init_allocation_ddpg, init_allocation_ddpg_order[0])
		utilities.write_results("Portfolio value in next period in exp " + str(exp_num) + " iteration "+str(eval_steps)+" ## " + str(sum(init_allocation_ddpg)) + " ## " + str(init_allocation_ddpg), sp_result_file)
		init_allocation = copy.copy(init_allocation_ddpg)
		obs = np.expand_dims(next_obs, 0)
		rewards.append(reward)
		eval_steps += 1
		time_step += 1

		if eval_steps >= max_eval_steps:
			done = True
			returns = np.exp(np.cumsum(rewards))
			all_rewards = returns if len(all_rewards) == 0 else np.vstack((all_rewards, returns))
			all_weights = np.array([current_weights]) if len(all_weights) == 0 else np.vstack((all_weights, [current_weights]))
			np.save(cur_work_dir + 'results/{}/Returns_{}_{}_{}_{}.npy'.format(experiment_file,result_file,stock_selction_number, alpha_cvar, experiment_number), all_rewards)
			np.save(cur_work_dir + 'results/{}/Weights_{}_{}_{}_{}.npy'.format(experiment_file,result_file,stock_selction_number, alpha_cvar, experiment_number), all_weights)


np.save(cur_work_dir + 'results/{}/Returns_{}_{}_{}_{}.npy'.format(experiment_file,result_file,stock_selction_number, alpha_cvar, experiment_number), all_rewards)
np.save(cur_work_dir + 'results/{}/Weights_{}_{}_{}_{}.npy'.format(experiment_file,result_file,stock_selction_number, alpha_cvar, experiment_number), all_weights)

