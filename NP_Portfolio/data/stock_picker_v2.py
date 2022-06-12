import math

from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
import sklearn.metrics



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


def mmd(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None):
    res = 0
    a = sk.metrics.pairwise.rbf_kernel(x,gamma=gamma)
    res += (np.sum(a) - np.trace(a)) / sp.special.comb(len(x),2)
    a = sk.metrics.pairwise.rbf_kernel(y,gamma=gamma)
    res += (np.sum(a) - np.trace(a)) / sp.special.comb(len(y),2)
    a = sk.metrics.pairwise.rbf_kernel(x,y,gamma=gamma)
    res -= np.sum(a) / len(x) / len(y) / 0.5
    return res


def filter_by_date_range(rets: pd.DataFrame, start: str, stop: str):
    return rets.loc[(rets.index >= start) & (rets.index < stop)]


def filter_by_most_profitable(rets: pd.DataFrame, k: int):
    columns = rets.mean(axis=0).sort_values(ascending=False).iloc[:k].index.tolist()
    return rets[columns]


def filter_by_most_volatile(rets: pd.DataFrame, k: int):
    columns = rets.std(axis=0).sort_values(ascending=False).iloc[:k].index.tolist()
    return rets[columns]


def filter_by_markowitz_model(rets: pd.DataFrame, k: int, rho=1):
    mu = rets.mean(axis=0).values
    sigma = rets.cov().values
    x = cp.Variable(len(mu), integer=True)
    ret = mu.T @ x
    risk = cp.quad_form(x,sigma)
    objective = cp.Maximize(ret - rho * risk)
    constraints = [cp.sum(x) == k, x >= 0, x <= 1]
    prob = cp.Problem(objective,constraints)
    prob.solve()
    columns = [name for name, i in zip(rets.columns,np.round(x.value)) if i == 1]
    return rets[columns]


def get_data_filter(
        num_stocks: int, 
        start_date: str, 
        end_date: str, 
        method: str = 'profitable', 
        top_k: int = 100, 
        rho: float = 1, 
        dirpath: str = 'sandp500_2008_2019',
    ):
    rets = load_returns(dirpath).dropna(axis=1)
    df = filter_by_date_range(rets, start_date, end_date)
    if method == 'profitable':
        df = filter_by_most_profitable(df, top_k)
    if method == 'volatile':
        df = filter_by_most_volatile(df, top_k)
    df = filter_by_markowitz_model(df, num_stocks, rho=rho)
    return df.columns.tolist()


if __name__ == '__main__':
    print(get_data_filter(20, '2008-01-01', '2018-01-01'))
