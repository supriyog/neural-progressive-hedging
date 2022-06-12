import numpy as np
import pandas as pd

class k_Markov_Generator(object):   
    
    def __init__(self,
                 asset_name,
                 k=30, # order of the Markov model
                 interval=15, # number of minutes to resample the raw data
                 start_date='2017-09-11',
                 end_date='2018-02-16',
                 data_loc='./sandp500/intraday.csv' # intraday data from https://www.kaggle.com/nickdl/snp-500-intraday-data
                 ):
        
        trading_min_per_day = 390
        self.asset_name = asset_name
        self.k = k
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.data_loc = data_loc
        self.n_trading_day_per_year = 253
        self.n_interval_per_day = int(trading_min_per_day/interval)
        self.no_asset = 2*len(self.asset_name)
        
        print('(synthetic) load the dataset...')
        self.start_pt, self.data = self.read_data()
        print('(synthetic) fit the model...')
        self.x_mean, self.sigma_ , self.sig_= self.fit_data()
        
    def read_data(self):
        
        dataset = pd.read_csv(self.data_loc, index_col=0, header=[0, 1]).sort_index(axis=1)
        temp_1 = dataset[self.asset_name]
        temp_2 = []
        for asset in self.asset_name:
            df = temp_1[asset][['close','volume']]
            df.columns = ['close_{}'.format(asset), 'volume_{}'.format(asset)]
            temp_2.append(df)
        temp_3 = pd.concat(temp_2, axis=1)
        mask = (temp_3.index>self.start_date) & (temp_3.index<self.end_date)
        temp_4 = temp_3.loc[mask]
              
        asset_raw_data = temp_4
        start_pt = asset_raw_data.iloc[-1]
        raw_data_ = asset_raw_data.interpolate(method='linear', axis=0).ffill().bfill()
        raw_data_.index = pd.to_datetime(raw_data_.index)
        raw_data_ = raw_data_.resample('{}min'.format(self.interval)).mean() 
        raw_data_ = raw_data_.dropna()
        raw_data_ = raw_data_.pct_change()+1
        raw_data_ = raw_data_.dropna()
        raw_data_ = np.log(raw_data_)
        raw_data = raw_data_.dropna()
        
        data_list = []
        for i in reversed(range(self.k)):
            temp = raw_data.iloc[i:-self.k+i]
            data_list.append(temp.reset_index(drop=True))
        data = pd.concat(data_list, axis=1)
        
        return start_pt, data
    
    def fit_data(self):
        
        _, data = self.read_data()
        x = data.values
        x_mean = x.mean(axis=0)
        cov = np.zeros((len(x_mean),len(x_mean)))
        for i in range(np.shape(x)[0]):
            cov = cov + np.outer(x[i] - x_mean, x[i] - x_mean)
        cov = cov/np.shape(x)[0]
        sigma_11 = cov[:self.no_asset,:self.no_asset]
        sigma_12 = cov[:self.no_asset,self.no_asset:]
        sigma_21 = cov[self.no_asset:,:self.no_asset]
        sigma_22 = cov[self.no_asset:,self.no_asset:]
        sig_ = np.dot(sigma_12, np.linalg.pinv(sigma_22))
        sigma_ = sigma_11 - np.dot(sig_, sigma_21)  
        
        return x_mean, sigma_, sig_
        
    def generate(self, n_years=1, seed=0):
        
        np.random.seed(seed)
        print('(synthetic) generate new data...')
        n_samples = int(self.n_interval_per_day*self.n_trading_day_per_year*n_years)
        data_new_pct = np.zeros((n_samples, self.no_asset))
        cond_sample = self.data.iloc[-1,:(self.k-1)*self.no_asset].values
        
        # generate the percentage change
        for i in range(n_samples):
            mu_ = self.x_mean[:self.no_asset] \
                + np.dot(self.sig_, cond_sample-self.x_mean[self.no_asset:])
            sample_ = np.random.multivariate_normal(mu_, self.sigma_)
            data_new_pct[i,:] = np.exp(sample_)
            cond_sample_new = np.concatenate([sample_, cond_sample[:-self.no_asset]])
            cond_sample = cond_sample_new
        
        # map the percentage back to absolute value
        generated_data = np.zeros([int(self.no_asset/2), self.n_trading_day_per_year*n_years, 5])
        # volume   
        data_volume = np.zeros([int(self.no_asset/2), n_samples])
        for i in range(int(self.no_asset/2)):
            asset_volume = self.start_pt.iloc[2*i+1]
            generated_data_pct_temp_volume = data_new_pct[:,2*i+1]
            for j in range(n_samples):
                volume_ = asset_volume * generated_data_pct_temp_volume[j]
                asset_volume = volume_
                data_volume[i, j] = volume_
        # price
        for i in range(int(self.no_asset/2)):
            asset_close = self.start_pt.iloc[2*i]
            for j in range(self.n_trading_day_per_year*n_years):
                generated_data_pct_temp = data_new_pct[j*self.n_interval_per_day:(j+1)*self.n_interval_per_day,2*i]
                generated_data_volume_temp = data_volume[i, j*self.n_interval_per_day:(j+1)*self.n_interval_per_day]
                generated_data[i, j, 0] = asset_close
                generated_data[i, j, 1] = generated_data_pct_temp.max()*asset_close
                generated_data[i, j, 2] = generated_data_pct_temp.min()*asset_close
                generated_data[i, j, 3] = generated_data_pct_temp[-1]*asset_close
                generated_data[i, j, 4] = generated_data_volume_temp.sum()
                asset_close = generated_data_pct_temp[-1]*asset_close
                
        return generated_data


if __name__ == '__main__':
    asset_name = ["AAPL", "ATVI", "CMCSA", "COST", "CSX", "DISH", "EA", "EBAY", "FB", 
                "GOOGL", "HAS", "ILMN", "INTC", "MAR", "SBUX"]
        
    generator = k_Markov_Generator(asset_name, k=30, start_date='2017-09-11', end_date='2018-02-16')
    new_data = generator.generate(n_years=5, seed=0)

    print(new_data)
    print(new_data.shape)

