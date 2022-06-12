import operator
import os
import sys
from pathlib import Path

import argparse
import numpy as np
import yaml

from pandas_datareader.data import DataReader
import pandas as pd
import universal as up
from universal import tools
from universal import algos
import logging

from utils import utilities


cur_work_dir = os.getcwd()+"/"

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument("--rf", type=str, default="result_UP_v1")
parser.add_argument("--ef", type=str, default="exp01")
parser.add_argument("--sn", type=str, default="0")
parser.add_argument("--en", type=str, default="0")
parser.add_argument("--an", type=str, default="0")
parser.add_argument("--uf", type=str, default="UPweights01")

args = parser.parse_args()
result_file = args.rf
experiment_file = args.ef
stock_selction_number = int(args.sn)
experiment_number = int(args.en)
algo_name_ID = int(args.an)
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
test_history = test_history[:,start_idx:,:]
test_data = (test_history, stocks)


train_price_vector = train_history
eval_price_vector = test_history

max_eval_steps = 30

algo_names = ["OLMAR", "PAMR", "RMR"]

if not os.path.exists(cur_work_dir+'experiments/'+upweights_dir+'/'):
    os.makedirs(cur_work_dir+'experiments/'+upweights_dir+'/')

stocks.insert(0,"cash")
np.set_printoptions(threshold=np.inf)
algo = algos.OLMAR(window = 5, eps = 10)
for exp_num in range(experiment_number, experiment_number+1):
	test_data = (test_history[:,max_eval_steps*exp_num:,:], stocks)
	print ('{}_{}_{}_{}'.format(experiment_file,algo_names[algo_name_ID],stock_selction_number,experiment_number))
	test_data_exp = test_history[:,max_eval_steps*exp_num:max_eval_steps*(exp_num+1):,3]
	test_data_exp_appended = np.array([[1.0 for cs in range(test_data_exp.shape[1])]])
	test_data_exp_appended = np.vstack((test_data_exp_appended, test_data_exp))
	test_data_UP = pd.DataFrame(np.swapaxes(test_data_exp_appended, 0, 1), columns = stocks)
	#print (test_data_UP)

	if(algo_name_ID == 0):
		algo = algos.OLMAR(window=5, eps=10)
	elif(algo_name_ID == 1):
		algo = algos.PAMR()
	elif(algo_name_ID == 2):
		algo = algos.RMR()

	algo_result = algo.run(test_data_UP)
	algo_result.fee = 0.002
	algo_result._recalculate()
	all_weights = algo_result.weights
	#all_weights = np.array([current_weights]) if len(all_weights) == 0 else np.vstack((all_weights, [current_weights]))

	np.save(cur_work_dir+'experiments/{}/UP_weights_{}_{}_{}.npy'.format(upweights_dir,algo_names[algo_name_ID],stock_selction_number,experiment_number), all_weights)

