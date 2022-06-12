from environment.GBM import GBM_Generator

def get_data(start_date, end_date, stocks, data_loc, interval=120, n_years=5, seed=0, idx_of_date=None, use_intraday=True):
    return GBM_Generator(
        asset_name=stocks, interval=interval, 
        start_date=start_date, end_date=end_date,
        data_loc=data_loc, use_intraday=use_intraday
    ).generate(n_years=n_years, seed=seed), None
