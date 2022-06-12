
from environment.k_Markov_generator import k_Markov_Generator

def get_data(start_date, end_date, stocks, data_loc, k=30, n_years=5, seed=0, idx_of_date=None):
    return k_Markov_Generator(
        stocks, k=k, start_date=start_date, 
        end_date=end_date, data_loc=data_loc
    ).generate(n_years=n_years, seed=seed), None
