import numpy as np
import pandas as pd
import warnings
import datetime
from .sp500_data_loader import get_data

warnings.filterwarnings("ignore", category=RuntimeWarning)


class GBM_Generator(object):   
    """
    A Geometric Brownian motion (GBM) (also known as exponential Brownian motion) is 
    a continuous-time stochastic process in which the logarithm of the randomly varying 
    quantity follows a Brownian motion (also called a Wiener process) with drift.
    """
    
    def __init__(self,
                 asset_name,
                 interval=120, # number of minutes to resample the raw data
                 start_date='2017-09-11',
                 end_date='2018-02-16',
                 use_intraday=True,
                 data_loc='../sandp500/dataset.csv' # intraday data from https://www.kaggle.com/nickdl/snp-500-intraday-data
                 ):
        
        trading_min_per_day = 390
        self.asset_name = asset_name
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.use_intraday = use_intraday
        self.data_loc = data_loc
        self.n_trading_day_per_year = 253
        self.n_asset = len(self.asset_name)
        if use_intraday:
            assert datetime.datetime.strptime(self.start_date, "%Y-%m-%d").date()\
            >= datetime.datetime.strptime('2017-09-11', "%Y-%m-%d").date() and \
            datetime.datetime.strptime(self.end_date, "%Y-%m-%d").date()\
            <= datetime.datetime.strptime('2018-02-16', "%Y-%m-%d").date(), \
            'Intraday data is only available from 2017-09-11 to 2018-02-16.'
            self.dt = interval/trading_min_per_day
            self.n_interval_per_day = int(trading_min_per_day/interval)
        else:
            self.dt = 1
            self.n_interval_per_day = 1
        print('Read the intraday data...')
        self.start_pt, self.raw_data = self.read_data()
        
    def read_data(self):
        
        if self.use_intraday:
            dataset = pd.read_csv(self.data_loc, index_col=0, header=[0, 1]).sort_index(axis=1)
            temp_1 = dataset[self.asset_name]
            temp_2 = []
            for asset in self.asset_name:
                df = temp_1[asset][['close','volume']]
                df.columns = ['close_{}'.format(asset), 'volume_{}'.format(asset)]
                temp_2.append(df)
            temp_3 = pd.concat(temp_2, axis=1)
            mask = (temp_3.index>self.start_date) & (temp_3.index<self.end_date)
            temp = temp_3.loc[mask]
            asset_raw_data = temp
            start_pt = asset_raw_data.iloc[-1]
            raw_data_ = asset_raw_data.interpolate(method='linear', axis=0).ffill().bfill()
            raw_data_.index = pd.to_datetime(raw_data_.index)
            raw_data_ = raw_data_.resample('{}min'.format(self.interval)).mean() 
            raw_data_ = raw_data_.dropna()
            raw_data_ = raw_data_.pct_change()
            raw_data = raw_data_.dropna()
        else:
            #import os
            #os.chdir("..")
            dataset, _ = get_data(self.start_date, self.end_date, self.asset_name, data_loc=self.data_loc)
            temp_2 = []
            for i in range(len(self.asset_name)):
                data_temp = dataset[i, :, [3,4]].T 
                df_temp = pd.DataFrame(data_temp, columns=['close_{}'.format(self.asset_name[i]), 
                                                           'volume_{}'.format(self.asset_name[i])])
                temp_2.append(df_temp)
            temp = pd.concat(temp_2, axis=1)           
            asset_raw_data = temp
            start_pt = asset_raw_data.iloc[-1]
            raw_data_ = asset_raw_data.interpolate(method='linear', axis=0).ffill().bfill()
            raw_data_.index = pd.to_datetime(raw_data_.index)
            raw_data_ = raw_data_.pct_change()
            raw_data = raw_data_.dropna()
        
        return start_pt, raw_data
    
    # Geometric Brownian Motion
    def GBM(self, N, S_0, mu, sigma, seed): 
        """Params: 
            S_0: initial stock price/volume
            mu: returns (drift coefficient)
            sigma: volatility (diffusion coefficient)
            W: brownian motion
            N: number of increments
        """
        np.random.seed(seed)
        t = np.arange(1, int(N) + 1)
        b = np.random.normal(0, 1, int(N)) 
        W = b.cumsum() # Brownian paths
        drift = (mu - 0.5 * sigma**2) * t
        diffusion = sigma * W
        S = np.array([S_0 * np.exp(drift + diffusion)]) 
        S = np.squeeze(S)
        S = np.hstack((np.array(S_0), S))
            
        return S
    
    # Log normal distribution for volume
    def log_normal(self, N, V_0, mu, sigma, seed):
        np.random.seed(seed)
        V = []
        V.append(V_0)
        V_prev = V_0
        for i in range(int(N)):
            temp = np.random.normal(mu, sigma)
            V_temp = V_prev * np.exp(temp)
            V_prev = V_temp
            V.append(max(V_temp,0))
            
        return np.array(V)
    
    def generate(self, n_years=1, seed=0):
        np.random.seed(seed)
        seeds = np.random.randint(500, size=self.n_asset) 
        T = int(self.n_interval_per_day*self.n_trading_day_per_year*n_years)
        N = T/self.dt
        data_price = np.zeros([self.n_asset, int(N)+1])
        data_volume = np.zeros([self.n_asset, int(N)+1])
        i = 0
        for asset in self.asset_name:
            print('Generate new data for {}...'.format(asset))
            # returns (drift coefficient)
            mu_price = self.raw_data['close_{}'.format(asset)].mean() 
            mu_volume = np.mean(np.log(self.raw_data['volume_{}'.format(asset)]+1))
            # volatility (diffusion coefficient)
            sigma_price = self.raw_data['close_{}'.format(asset)].std() 
            sigma_volume = np.std(np.log(self.raw_data['volume_{}'.format(asset)]+1))
            data_price[i, :] = self.GBM(N, S_0=self.start_pt['close_{}'.format(asset)], mu=mu_price, \
                      sigma=sigma_price, seed=seeds[i])
            data_volume[i, :] = self.log_normal(N, V_0=self.start_pt['volume_{}'.format(asset)], \
                       mu=mu_volume, sigma=sigma_volume, seed=seeds[i])
            i += 1
        generated_data = np.zeros([self.n_asset, self.n_trading_day_per_year*n_years, 5])
        
        if self.use_intraday: 
            for i in range(self.n_asset):
                asset_close = self.start_pt.iloc[2*i]
                for j in range(self.n_trading_day_per_year*n_years):
                    generated_data_price_temp = data_price[i, j*self.n_interval_per_day:(j+1)*self.n_interval_per_day]
                    generated_data_volume_temp = data_volume[i, j*self.n_interval_per_day:(j+1)*self.n_interval_per_day]
                    generated_data[i, j, 0] = asset_close
                    generated_data[i, j, 1] = generated_data_price_temp.max()
                    generated_data[i, j, 2] = generated_data_price_temp.min()
                    generated_data[i, j, 3] = generated_data_price_temp[-1]
                    generated_data[i, j, 4] = generated_data_volume_temp.sum()
                    asset_close = generated_data_price_temp[-1]
        else:
            for i in range(self.n_asset):   
                asset_close = self.start_pt.iloc[2*i]       
                for j in range(self.n_trading_day_per_year*n_years):
                    generated_data[i, j, 0] = asset_close
                    generated_data[i, j, 1] = data_price[i, j]
                    generated_data[i, j, 2] = data_price[i, j]
                    generated_data[i, j, 3] = data_price[i, j]
                    generated_data[i, j, 4] = data_volume[i, j]
                    asset_close = data_price[i, j]
                            
        return generated_data
    
if __name__ == '__main__':
    
    asset_name = ["AAPL", "ATVI", "CMCSA", "COST", "CSX", "DISH", "EA", "EBAY", "FB", 
                  "GOOGL", "HAS", "ILMN", "INTC", "MAR", "SBUX"]   
    
    generator = GBM_Generator(asset_name=asset_name, start_date='2005-01-01',
                 end_date='2016-12-31', use_intraday=False, data_loc='sandp500_2005_2019/{}_data.csv')
    
    generated_data = generator.generate(n_years=1, seed=0)

    

