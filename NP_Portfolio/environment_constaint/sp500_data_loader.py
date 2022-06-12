import pandas as pd
import  numpy as np
import datetime as dt

def get_data(start_date, end_date, stocks, data_loc='data/sandp500/individual_stocks_5yr/individual_stocks_5yr/{}_data.csv', idx_of_date=None, return_dates=False):
    #print('Get sp500 data')
    #data_loc='data/sandp500_2005_2019/{}_data.csv'

    # ref_date = dt.date(2013, 2, 8)
    ref_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()

    abbr = []
    feats = []
    days = []
    for symb in stocks:
        x = pd.read_csv(data_loc.format(symb))
        mask = (x['date'] >= start_date) & (x['date'] < end_date)
        x = x.loc[mask]
        z = {}
        z['day'] = np.array([(dt.datetime.strptime(s, '%Y-%m-%d').date() - ref_date).days for s in x['date']])
        z['open'] = x['open'].values
        z['high'] = x['high'].values
        z['low'] = x['low'].values
        z['close'] = x['close'].values
        z['volume'] = x['volume'].values
        f = np.array([z['day'], z['open'], z['high'], z['low'], z['close'], z['volume']]).T
        if np.isnan(f).any():
            print('skip file with NaN:', symb)
            continue
        feats.append(f)
        abbr.append(symb)
        if len(days) > 0:
            days = np.intersect1d(days, z['day'])
        else:
            days = z['day']
            #print('Excluding {}'.format(symb))
        #print('{} {}'.format(symb, len(z['day'])))

    if idx_of_date is not None:
        idx_of_date = (dt.datetime.strptime(idx_of_date, '%Y-%m-%d').date() - ref_date).days
        idx_of_date = int(np.argmax(days >= idx_of_date))

    prices = []
    for f in feats:
        idx = np.nonzero(np.isin(f[:, 0], days))[0]
        prices.append(f[idx, 1:])

    if return_dates:
        dates = [ref_date + dt.timedelta(days=float(d)) for d in days]
        return np.array(prices), idx_of_date, dates
    else:
        return np.array(prices), idx_of_date
