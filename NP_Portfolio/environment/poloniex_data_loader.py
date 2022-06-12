import numpy as np
import pandas as pd

def get_data(start_date, end_date, stocks, data_loc='poloniex_30m.hf', idx_of_date=None, return_dates=False):
    df1 = pd.read_hdf(data_loc, key='train')
    df2 = pd.read_hdf(data_loc, key='test')
    df = pd.concat([df1,df2], axis=0)
    df.replace(np.nan, 0, inplace=True)
    df = df.fillna(method='pad')
    mask = (df.index >= start_date) & (df.index < end_date)
    abbr = []
    feats = []
    for symb in stocks:
        x = df.loc[mask]
        z = {}
        z['open'] = x[(symb,'open')].values
        z['high'] = x[(symb,'high')].values
        z['low'] = x[(symb,'low')].values
        z['close'] = x[(symb,'close')].values
        z['volume'] = x[(symb,'close')].values
        f = np.array([z['open'], z['high'], z['low'], z['close'], z['volume']]).T
        feats.append(f)
        abbr.append(symb)

    if idx_of_date is not None:
        idx_of_date = int(np.argmax(df.loc[mask].index >= idx_of_date))

    prices = []
    for f in feats:
        prices.append(f[:,:])

    if return_dates:
        dates = df.loc[mask].index
        return np.array(prices), idx_of_date, dates
    else:
        return np.array(prices), idx_of_date

