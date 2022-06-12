import copy
import operator
import sys

import numpy as np

mod = sys.modules[__name__]

import environment.sp500_data_loader as sp500_data_loader
import environment.kmm_data_loader as kmm_data_loader
import environment.gbm_data_loader as gbm_data_loader
import environment.poloniex_data_loader as poloniex_data_loader


def get_vermouth_training_data_config():
    return {'start_date': '2013-02-08',
            'end_date': '2016-12-30',
            'data_loc': 'data/sandp500/individual_stocks_5yr/individual_stocks_5yr/{}_data.csv'}

def get_vermouth_validation_data_config():
    return {'start_date': '2013-02-08',
            'end_date': '2018-02-16',
            'idx_of_date': '2017-01-03',
            'data_loc': 'data/sandp500/individual_stocks_5yr/individual_stocks_5yr/{}_data.csv'}

def get_vermouth_testing_data_config():
    return {'start_date': '2013-02-08',
            'end_date': '2018-02-16',
            'idx_of_date': '2017-01-03',
            'data_loc': 'data/sandp500/individual_stocks_5yr/individual_stocks_5yr/{}_data.csv'}

def get_expanded_training_data_config():
    return {'start_date': '2008-01-01',
            'end_date': '2016-01-01',
            'data_loc': 'data/sandp500_2008_2019/{}_data.csv'}

def get_expanded_validation_data_config():
    return {'start_date': '2008-01-01',
            'end_date': '2018-01-01',
            'idx_of_date': '2016-01-01',
            'data_loc': 'data/sandp500_2008_2019/{}_data.csv'}

def get_expanded_testing_data_config():
    return {'start_date': '2008-01-01',
            'end_date': '2020-01-01',
            'idx_of_date': '2018-01-01',
            'data_loc': 'data/sandp500_2008_2019/{}_data.csv'}

def get_pq_training_data_config():
    return {'start_date': '2005-01-01',
            'end_date': '2016-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_pq_training_post_crisis_data_config():
    return {'start_date': '2009-07-01',
            'end_date': '2016-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_pq_validation_data_config():
    return {'start_date': '2005-01-01',
            'end_date': '2017-01-01',
            'idx_of_date': '2016-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_pq_training_and_validation_post_crisis_data_config():
    return {'start_date': '2009-07-01',
            'end_date': '2017-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_pq_testing_data_config():
    return {'start_date': '2005-01-01',
            'end_date': '2018-12-04',
            'idx_of_date': '2017-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_synthetic_kmm_training_data_config():
    return {'start_date': '2017-09-11',
            'end_date': '2018-02-16',
            'k': 30, 'n_years': 8, 'seed': 0,
            'data_loc': 'data/sandp500/intraday.csv'}

def get_synthetic_kmm_validation_data_config():
    return {'start_date': '2017-09-11',
            'end_date': '2018-02-16',
            'k': 30, 'n_years': 2, 'seed': 1,
            'data_loc': 'data/sandp500/intraday.csv'}

def get_synthetic_kmm_testing_data_config():
    return {'start_date': '2017-09-11',
            'end_date': '2018-02-16',
            'k': 30, 'n_years': 2, 'seed': 2,
            'data_loc': 'data/sandp500/intraday.csv'}

def get_synthetic_gbm_training_data_config():
    return {'start_date': '2017-09-11',
            'end_date': '2018-02-16',
            'interval': 120, 'n_years': 8, 'seed': 0,
            'data_loc': 'data/sandp500/intraday.csv'}

def get_synthetic_gbm_validation_data_config():
    return {'start_date': '2017-09-11',
            'end_date': '2018-02-16',
            'interval': 120, 'n_years': 2, 'seed': 1,
            'data_loc': 'data/sandp500/intraday.csv'}

def get_synthetic_gbm_testing_data_config():
    return {'start_date': '2017-09-11',
            'end_date': '2018-02-16',
            'interval': 120, 'n_years': 2, 'seed': 2,
            'data_loc': 'data/sandp500/intraday.csv'}

def get_synthetic_daily_gbm_training_data_config():
    return {'start_date': '2005-01-01',
            'end_date': '2017-01-01',
            'interval': 120, 'n_years': 8, 'seed': 0, 'use_intraday': False,
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_synthetic_daily_gbm_validation_data_config():
    return {'start_date': '2005-01-01',
            'end_date': '2017-01-01',
            'interval': 120, 'n_years': 2, 'seed': 1, 'use_intraday': False,
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_synthetic_daily_gbm_testing_data_config():
    return {'start_date': '2005-01-01',
            'end_date': '2017-01-01',
            'interval': 120, 'n_years': 2, 'seed': 2, 'use_intraday': False,
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_full_v1_training_data_config():
    return {'start_date': '2005-01-01',
            'end_date': '2017-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_full_v1_validation_data_config():
    return {'start_date': '2005-01-01',
            'end_date': '2018-01-01',
            'idx_of_date': '2017-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_full_v1_training_and_validation_data_config():
    return {'start_date': '2005-01-01',
            'end_date': '2018-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_full_v1_testing_data_config():
    return {'start_date': '2005-01-01',
            'end_date': '2020-01-01',
            'idx_of_date': '2018-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_full_v2_training_data_config():
    return {'start_date': '2009-07-01',
            'end_date': '2017-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_full_v2_validation_data_config():
    return {'start_date': '2009-07-01',
            'end_date': '2018-01-01',
            'idx_of_date': '2017-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_full_v2_training_and_validation_data_config():
    return {'start_date': '2009-07-01',
            'end_date': '2018-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_full_v2_testing_data_config():
    return {'start_date': '2009-07-01',
            'end_date': '2020-01-01',
            'idx_of_date': '2018-01-01',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_ye_training_data_config():
    return {'start_date': '2006-10-20',
            'end_date': '2012-11-20',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_ye_testing_data_config():
    return {'start_date': '2006-10-20',
            'end_date': '2013-11-20',
            'idx_of_date': '2012-11-20',
            'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}

def get_poloniex_training_data_config():
    return {'start_date': '2015-06-30',
            'end_date': '2017-04-30',
            'data_loc': 'data/poloniex_30m.hf'}

def get_poloniex_testing_data_config():
    return {'start_date': '2015-06-30',
            'end_date': '2017-06-30',
            'idx_of_date': '2017-04-30',
            'data_loc': 'data/poloniex_30m.hf'}


def parse(desc, stocks):
    if desc == 'vermouth_training':
        data_config = get_vermouth_training_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'vermouth_validation':
        data_config = get_vermouth_validation_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'vermouth_testing':
        data_config = get_vermouth_testing_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'expanded_training':
        data_config = get_expanded_training_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'expanded_validation':
        data_config = get_expanded_validation_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'expanded_testing':
        data_config = get_expanded_testing_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'pq_training':
        data_config = get_pq_training_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'pq_training_post_crisis':
        data_config = get_pq_training_post_crisis_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'pq_validation':
        data_config = get_pq_validation_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'pq_training_and_validation_post_crisis':
        data_config = get_pq_training_and_validation_post_crisis_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'pq_testing':
        data_config = get_pq_testing_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'kmm_training':
        data_config = get_synthetic_kmm_training_data_config()
        data_loader = 'k_kmm_data_loader.get_data'
    elif desc == 'kmm_validation':
        data_config = get_synthetic_kmm_validation_data_config()
        data_loader = 'k_kmm_data_loader.get_data'
    elif desc == 'kmm_testing':
        data_config = get_synthetic_kmm_testing_data_config()
        data_loader = 'k_kmm_data_loader.get_data'
    elif desc == 'gbm_training':
        data_config = get_synthetic_gbm_training_data_config()
        data_loader = 'gbm_data_loader.get_data'
    elif desc == 'gbm_validation':
        data_config = get_synthetic_gbm_validation_data_config()
        data_loader = 'gbm_data_loader.get_data'
    elif desc == 'gbm_testing':
        data_config = get_synthetic_gbm_testing_data_config()
        data_loader = 'gbm_data_loader.get_data'
    elif desc == 'daily_gbm_training':
        data_config = get_synthetic_daily_gbm_training_data_config()
        data_loader = 'gbm_data_loader.get_data'
    elif desc == 'daily_gbm_validation':
        data_config = get_synthetic_daily_gbm_validation_data_config()
        data_loader = 'gbm_data_loader.get_data'
    elif desc == 'daily_gbm_testing':
        data_config = get_synthetic_daily_gbm_testing_data_config()
        data_loader = 'gbm_data_loader.get_data'
    elif desc == 'full_v1_training':
        data_config = get_full_v1_training_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'full_v1_validation':
        data_config = get_full_v1_validation_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'full_v1_training_and_validation':
        data_config = get_full_v1_training_and_validation_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'full_v1_testing':
        data_config = get_full_v1_testing_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'full_v2_training':
        data_config = get_full_v2_training_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'full_v2_validation':
        data_config = get_full_v2_validation_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'full_v2_training_and_validation':
        data_config = get_full_v2_training_and_validation_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'full_v2_testing':
        data_config = get_full_v2_testing_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'ye_training':
        data_config = get_ye_training_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'ye_testing':
        data_config = get_ye_testing_data_config()
        data_loader = 'sp500_data_loader.get_data'
    elif desc == 'poloniex_training':
        data_config = get_poloniex_training_data_config()
        data_loader = 'poloniex_data_loader.get_data'
    elif desc == 'poloniex_testing':
        data_config = get_poloniex_testing_data_config()
        data_loader = 'poloniex_data_loader.get_data'
    elif desc.startswith('custom'):
        data_config = {'start_date': desc.split('_')[1],
                       'end_date': desc.split('_')[2],
                       'data_loc': 'data/sandp500_2005_2019/{}_data.csv'}
        if len(desc.split('_')) > 3:
            data_config['idx_of_date'] = desc.split('_')[3]
        data_loader = 'sp500_data_loader.get_data'
    else:
        raise
    data_config['stocks'] = copy.deepcopy(stocks)
    return data_loader, data_config


def get_data(loader, config):
    return operator.attrgetter(loader)(mod)(**config)
