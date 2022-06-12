from pathlib import Path

import pandas as pd
import numpy as np

from . import stock_loading
from . import data_loading

stock_loader, stock_config = stock_loading.parse('multi_tech')
stock_config['mode'] = 'training'
stock_config['num_stocks'] = 3
stock_config['sample_idx'] = 99
stock_config['split_seed'] = 0
stock_config['sampling_seed'] = None
#stocks = stock_loading.get_stocks(stock_loader, stock_config)

#print(stocks)
#print(len(stocks))

#data_loader, data_config = data_loading.parse('full_v1_training', stocks)
#data_config['start_date'] = '2007-01-01'
#data_config['end_date'] = '2017-01-01'
#history, idx_of_date = data_loading.get_data(data_loader, data_config)
##
#print(history[0].shape)
#print(idx_of_date)
#print(len(stocks))
#
#data_loader, data_config = data_loading.parse('full_v1_validation', stocks)
#history, idx_of_date = data_loading.get_data(data_loader, data_config)
#
#data_loader, data_config = data_loading.parse('full_v1_testing', stocks)
#history, idx_of_date = data_loading.get_data(data_loader, data_config)

#print(stocks)
#
#
#prng = np.random.RandomState(0)
#stocks = prng.choice(stocks, size=50, replace=False)
#print(stocks)
#

#prng = np.random.RandomState(0)

total = 0

for k in range(30):
    stock_config['num_stocks'] = 9
    stock_config['sample_idx'] = k
    stocks = stock_loading.get_stocks(stock_loader, stock_config)
    print(stocks)
    data_loader, data_config = data_loading.parse('full_v1_training', stocks)
    data_config['start_date'] = '2007-01-01'
    data_config['end_date'] = '2017-01-01'
    history, idx_of_date = data_loading.get_data(data_loader, data_config)
    print(history.shape[1])
    total += history.shape[1]

print(total)
