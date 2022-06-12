import copy
import itertools
import os
import shutil
import stat

from pathlib import Path

from ray.rllib.agents.trainer import COMMON_CONFIG
import yaml

from . import utils
from data import stock_loading
from data import data_loading

from . import ray_default


def setup_exp(dirpath, stock_descs, data_desc, base_env_config, 
              rllib_trainer_config, rllib_modules, summary_desc=None):
    if dirpath.exists():
        raise Exception('The directory {} already exists.'.format(dirpath))
    dirpath.mkdir()
    universes = {}
    for universe_name,stock_desc in stock_descs.items():
        if isinstance(stock_desc,str):
            stock_loader, stock_config = stock_loading.parse(stock_desc)
            universes[universe_name] = {'stock_config': stock_config,
                                        'stock_loader': stock_loader,
                                        'stocks': stock_loading.get_stocks(stock_loader, stock_config)}
        else:
            universes[universe_name] = {'stock_config': None,
                                        'stock_loader': None,
                                        'stocks': stock_desc}
    rllib_env_configs = {}
    for universe_name in universes:
        data_loader, data_config = data_loading.parse(data_desc, universes[universe_name]['stocks'])
        rllib_env_configs[universe_name] = copy.deepcopy(base_env_config)
        rllib_env_configs[universe_name]['stock_config'] = universes[universe_name]['stock_config']
        rllib_env_configs[universe_name]['stock_loader'] = universes[universe_name]['stock_loader']
        rllib_env_configs[universe_name]['data_config'] = data_config
        rllib_env_configs[universe_name]['data_loader'] = data_loader
    if summary_desc:
        with (dirpath / 'README.txt').open('w') as f:
            f.write(summary_desc)
    with (dirpath / 'base_env_config.yaml').open('w') as f:
        yaml.dump(base_env_config, f)
    with (dirpath / 'rllib_env_config.yaml').open('w') as f:
        yaml.dump(rllib_env_configs, f)
    with (dirpath / 'rllib_trainer_config.yaml').open('w') as f:
        yaml.dump(rllib_trainer_config, f)
    for module in rllib_modules:
        shutil.copyfile(str(module), str(dirpath / module.name))


if __name__ == '__main__':
    stock_desc = {}
    for k in range(30):
        name = 'multi_tech_training_9_{}'.format(k)
        stock_desc[name] = name
    base_env_config = {
        'base_env_params': {
            'steps': 0,
            'window_length': 30,
            'trading_cost': 0.0020,
            'init_weights': 'eq',
        }
    }
    rllib_trainer_config = COMMON_CONFIG.copy()
    rllib_trainer_config["num_gpus"] = 1
    rllib_trainer_config["num_workers"] = 0
    rllib_trainer_config["learning_rate"] = 2e-5
    rllib_trainer_config["timesteps_per_iteration"] = 3000
    rllib_trainer_config["train_batch_size"] = 50
    rllib_trainer_config["replay_bias"] = 5e-4 #5e-5 or 5e-4 or 1e-3
    rllib_trainer_config["dump_gradients"] = None
    rllib_trainer_config["custom_policy_config"] = {
        'trading_cost': base_env_config['base_env_params']['trading_cost'],
        'separate_cash': True,
        'cash_bias_trainable': True,
        'predictor_type': 'rnn',
        'predictor_hiddens': [25],
        'predictor_filters': [None],
        'predictor_regularizer_weights': [0],
        'final_regularizer_weight': 1e-8,
        'rebalance_cash': True,
        'activation': 'relu',
    }
    rllib_modules = [
        Path('experiments') / 'custom_multi_dpm_continuous' / 'custom_policy.py',
        Path('experiments') / 'custom_multi_dpm_continuous' / 'custom_replay_buffer.py',
        Path('experiments') / 'custom_multi_dpm_continuous' / 'custom_optimizer.py',
        Path('experiments') / 'custom_multi_dpm_continuous' / 'custom_trainer.py',
    ]
    summary_desc = """
"""
    max_steps = 15e5
    
    settings = list(itertools.product(
        [
            ('custom_2006-01-01_2016-01-01'),
            ('custom_2007-01-01_2017-01-01'),
            ('custom_2008-01-01_2018-01-01'),
            ('custom_2009-01-01_2019-01-01'),
        ],
        [
            25,
        ]))
    gpus = [0,1,0,1]
    runfile = None
    for k, ((data_desc, units), gpu) in enumerate(zip(settings,gpus)):
        exp_name = 'exp{}'.format(4001+k)
        dirpath = Path('experiments').resolve() / exp_name
        rllib_trainer_config["custom_policy_config"]["predictor_hiddens"] = [units]
        setup_exp(dirpath, stock_desc, data_desc, base_env_config, 
                rllib_trainer_config, rllib_modules, summary_desc=summary_desc)
        cmd = ' '.join(['python', '-m', 'training.train_multi_env', exp_name, 
            '-m', '{:d}'.format(int(max_steps)),
            '-p', '{:d}'.format(int(ray_default.port)),
            #'-g', '{:d}'.format((int(exp_name[3:])+1) % 2),
            '-g', '{:d}'.format(gpu),
            '-f', '1D',
            '>', 'experiments/{}/out.txt'.format(exp_name)])
        runfile = Path('training') / 'exp{}.sh'.format(exp_name[3:])
        with runfile.open('w') as f:
            f.write(cmd)
            f.close()
        os.chmod(runfile, os.stat(runfile).st_mode | stat.S_IEXEC)
