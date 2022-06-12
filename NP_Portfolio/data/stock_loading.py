import csv
import operator
import sys

import numpy as np

mod = sys.modules[__name__]

def get_vermouth_stocks():
    # vermouth stocks excluding REGN (not sure why we excluded this)
    return ['AAPL', 'ATVI', 'CMCSA', 'COST', 'CSX', 
            'DISH', 'EA', 'EBAY', 'FB', 'GOOGL', 
            'HAS', 'ILMN', 'INTC', 'MAR', 'SBUX']

def get_pq_stocks():
    return ['COST', 'CSCO', 'F', 'GS', 'AIG', 'AMGN'] # excludes 'CAT'

def get_top_gain_stocks(num_stocks, start_date, end_date):
    # shiau hong's pick based on top historical gain over specified period
    # only works over vermouth period, i.e. from 2013-02-08 to 2018-02-16
    from . import stock_picker
    return stock_picker.get_data_filter(num_stocks, start_date, end_date)

def get_markowitz_most_profitable_stocks(num_stocks, start_date, end_date, top_k=100, rho=1, dirpath='sandp500_2008_2019'):
    from . import stock_picker_v2
    return stock_picker_v2.get_data_filter(
        num_stocks, start_date, end_date, 'profitable', top_k, rho, dirpath)

def get_markowitz_most_volatile_stocks(num_stocks, start_date, end_date, top_k=100, rho=1, dirpath='sandp500_2008_2019'):
    from . import stock_picker_v2
    return stock_picker_v2.get_data_filter(
        num_stocks, start_date, end_date, 'volatile', top_k, rho, dirpath)

def get_random_stocks(sample_idx):
    with open('data/stock_universes.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for k, row in enumerate(reader):
            if k == sample_idx:
                return row

def get_k_random_stocks(num_stocks, sample_idx):
    from . import stock_picker_random
    stocks, stats = stock_picker_random.get_samples(num_stocks, sample_idx+1, '2005-01-01', '2020-01-01', 'data/sandp500_2005_2019')
    return [str(s) for s in stocks[sample_idx]]

def get_ye_stocks():
    return ['GOOG', 'NVDA', 'AMZN', 'AMD', 'QCOM', 'INTC', 'MSFT', 'AAPL'] # excludes 'BIDU'

def get_poloniex_stocks():
    return ['DASHBTC', 'LTCBTC', 'XMRBTC']

def get_multi_tech_stocks(mode='all', num_stocks=None, sample_idx=0, split_seed=0, sampling_seed=0):
    stocks = ['AAPL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ADS', 'ADSK', 'AKAM', 
            #'AMAT', 
            'AMD',
            #'ANET', 
            'ANSS', 'APH', 
            'AVGO',  # late entrant
            #'BR', 
            'CDNS', 
            #'CDW', 
            'CRM', 'CSCO', 'CTSH',
            'CTXS', 'DXC', 'FFIV', 'FIS', 'FISV', 'FLIR', 
            #'FLT', 
            #'FTNT', 
            'GLW', 'GPN',
            #'HPE', # late entrant
            'HPQ', 'IBM', 'INTC', 'INTU', 
            #'IPGP', 
            'IT', 
            #'JKHY', 
            'JNPR', 
            #'KEYS',
            'KLAC', 
            #'LDOS', 
            'LRCX', 
            'MA', # late entrant
            'MCHP', 'MSFT', 'MSI', 
            'MU', # late entrant
            #'MXIM', 
            #'NLOK',
            #'NOW', 
            'NTAP', 'NVDA', 'ORCL', 
            #'PAYC', 
            'PAYX', 
            #'PYPL', # late entrant
            'QCOM', 
            #'QRVO', # late entrant
            'SNPS',
            'STX', 'SWKS', 
            'TEL', # late entrant
            #'TXN', 
            'V', 'VRSN', 'WDC', 
            'WU', # late entrant
            'XLNX', 'XRX', 
            #'ZBRA',
            'ATVI', 
            'CHTR', # late entrant
            'CMCSA', 'CTL', 'DIS', 
            'DISCK', # late entrant
            'DISH', 'EA', 
            'FB', # late entrant 
            #'FOX', # too late entrant
            'GOOG', 'IPG', 
            #'LYV', 
            'NFLX', 
            'NWS', # late entrant
            'OMC', 'T', 
            #'TMUS', 
            #'TTWO', 
            #'TWTR', 
            #'VIAC', 
            'VZ',
            ]
    if mode == 'all':
        stocks = stocks
    elif mode == 'training':
        if split_seed is None:
            stocks = stocks[:50]
        else:
            prng = np.random.RandomState(split_seed)
            stocks = prng.choice(stocks, size=50, replace=False).tolist()
    elif mode == 'testing':
        if split_seed is None:
            stocks = stocks[50:]
        else:
            prng = np.random.RandomState(split_seed)
            exclude = set(prng.choice(stocks, size=50, replace=False))
            stocks = [s for s in stocks if s not in exclude]
    if num_stocks is None:
        return stocks
    else:
        if sampling_seed is None:
            start = (num_stocks * sample_idx) % len(stocks)
            stop = start + num_stocks
            out = (stocks + stocks)[start:stop]
        else:
            prng = np.random.RandomState(sampling_seed)
            for _ in range(sample_idx+1):
                out = prng.choice(stocks, size=num_stocks, replace=False).tolist()
        return out


def parse(desc):
    if desc == 'vermouth':
        stock_loader = 'get_vermouth_stocks'
        stock_config = {}
    elif desc == 'pq':
        stock_loader = 'get_pq_stocks'
        stock_config = {}
    elif desc.startswith('gain'):
        num_stocks = int(stocks.split('_')[1])
        stock_loader = 'get_top_gain_stocks'
        stock_config = {'num_stocks': num_stocks, 
                        'start_date': '2013-02-08', 
                        'end_date': '2016-12-30'}
    elif desc.startswith('profitable'):
        num_stocks = int(stocks.split('_')[1])
        top_k = int(stocks.split('_')[2])
        stock_loader = 'get_markowitz_most_profitable_stocks'
        stock_config = {'num_stocks': num_stocks, 
                        'start_date': '2008-01-01', 
                        'end_date': '2018-01-01',
                        'top_k': top_k, 'rho': 1, 
                        'dirpath': 'sandp500_2008_2019'}
    elif desc.startswith('volatile'):
        num_stocks = int(stocks.split('_')[1])
        top_k = int(stocks.split('_')[2])
        stock_loader = 'get_markowitz_most_volatile_stocks'
        stock_config = {'num_stocks': num_stocks, 
                        'start_date': '2008-01-01', 
                        'end_date': '2018-01-01',
                        'top_k': top_k, 'rho': 1, 
                        'dirpath': 'sandp500_2008_2019'}
    elif desc.startswith('random'):
        sample_idx = int(desc.split('_')[-1])
        stock_loader = 'get_random_stocks'
        stock_config = {'sample_idx': sample_idx}
    elif desc.startswith('k_random'):
        num_stocks = int(desc.split('_')[-2])
        sample_idx = int(desc.split('_')[-1])
        stock_loader = 'get_k_random_stocks'
        stock_config = {'num_stocks': num_stocks,
                        'sample_idx': sample_idx}
    elif desc == 'ye':
        stock_loader = 'get_ye_stocks'
        stock_config = {}
    elif desc == 'poloniex':
        stock_loader = 'get_poloniex_stocks'
        stock_config = {}
    elif desc.startswith('multi_tech'):
        mode = desc.split('_')[2]
        num_stocks = int(desc.split('_')[3])
        sample_idx = int(desc.split('_')[4])
        stock_loader = 'get_multi_tech_stocks'
        stock_config = {'mode': mode,
                        'num_stocks': num_stocks,
                        'sample_idx': sample_idx}
    else:
        raise
    return stock_loader, stock_config

def get_stocks(loader, config):
    return operator.attrgetter(loader)(mod)(**config)
