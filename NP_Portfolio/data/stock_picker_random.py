from pathlib import Path

import numpy as np
import pandas as pd


def load_returns(dirpath: str, include_index: bool = False):
    dirpath = Path(dirpath)
    symb = '^GSPC'
    filepath = dirpath / '{}_data.csv'.format(symb)
    df = pd.read_csv(str(filepath))
    data = pd.DataFrame(df[['close']].values, index=df['date'].values, columns=[symb])
    for filepath in list(dirpath.glob('*.csv')):
        symb = filepath.name[:-9]
        if symb == '^GSPC':
            continue
        filepath = dirpath / '{}_data.csv'.format(symb)
        df = pd.read_csv(str(filepath))
        s = pd.Series(df['close'].values, index=df['date'].values, name=symb)
        if not s.isna().any():
            data = data.merge(s.to_frame(), how='outer', left_index=True, right_index=True)
    data = data.drop(labels='^GSPC', axis=1)
    return ((data - data.shift(periods=1)) / data.shift(periods=1)).iloc[1:,:]


def filter_by_date_range(rets: pd.DataFrame, start: str, stop: str):
    return rets.loc[(rets.index >= start) & (rets.index < stop)]


def get_samples(
        num_stocks: int, 
        num_samples: int, 
        start_date: str, 
        end_date: str, 
        dirpath: str = 'sandp500_2005_2019',
        seed: int = 0,
    ):
    rets = load_returns(dirpath).dropna(axis=1)
    df = filter_by_date_range(rets, start_date, end_date)
    all_stocks = list(df.columns)
    print('Total Number of Stocks = {}'.format(len(all_stocks)))
    prng = np.random.RandomState(seed)
    stocks = []
    stats = []
    for _ in range(num_samples):
        selected_stocks = list(prng.choice(all_stocks, size=num_stocks, replace=False))
        stocks.append(selected_stocks)
        selected_rets = rets[selected_stocks]
        selected_rets['Cash'] = np.zeros(len(selected_rets))
        selected_rets = selected_rets[['Cash'] + selected_stocks]
        mu = selected_rets.mean(axis=0).values
        sigma = selected_rets.cov().values
        non_cash_mu = mu[1:]
        stats.append([
                252 * np.mean(mu),
                np.sqrt(252) * np.sqrt(np.sum(sigma)/(num_stocks+1)/(num_stocks+1)),
                np.sqrt(252) * np.mean(mu) / np.sqrt(np.sum(sigma)/(num_stocks+1)/(num_stocks+1))
            ] +
            selected_stocks + 
            list(252 * non_cash_mu) +
            list(252 * non_cash_mu[np.argsort(non_cash_mu)])[::-1]
        )
    columns = ['CRP_ret', 'CRP_std', 'CRP_sharpe'] + \
        ['stock_{}'.format(k) for k in range(1,num_stocks+1)] + \
        ['stock_{}_ret'.format(k) for k in range(1,num_stocks+1)] + \
        ['ret_{}'.format(k) for k in range(1,num_stocks+1)]
    stats = pd.DataFrame(stats, columns=columns)
    return stocks, stats


if __name__ == '__main__':
    stocks, stats = get_samples(9, 1000, '2005-01-01', '2020-01-01')
    print(stocks)
    import csv
    with open('stock_universes.csv','w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in stocks:
            writer.writerow(row)
    stats.to_csv('stock_universe_stats.csv')
