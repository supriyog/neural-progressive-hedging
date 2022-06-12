import argparse
import copy
import datetime
import importlib
import operator
import os
import shutil
import sys
import time

from pathlib import Path

import numpy as np
import ray
import ray.rllib.agents.ddpg as ddpg
import yaml

from ray.tune.registry import register_env
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from . import utils

# ========== training ========== #

def train(rllib_env_config, rllib_trainer_config, rllib_trainer_cls,
          policies_dirpath, max_steps=1e6, env_cls=None):
    register_env("portfolio_env", lambda env: utils.load_env(rllib_env_config, env_cls))
    trainer = rllib_trainer_cls(config=rllib_trainer_config, env="portfolio_env")
    policies_dirpath.mkdir()
    print('tensorboard --logdir={}'.format(trainer.logdir), flush=True)
    start = time.time()
    steps = 0
    policy = trainer.workers.local_worker().policy_map[DEFAULT_POLICY_ID]
    policy.export_checkpoint(str(policies_dirpath), str(steps))
    while steps < max_steps:
        result = trainer.train()
        trainer.save()
        steps = trainer.optimizer.num_steps_sampled
        policy = trainer.workers.local_worker().policy_map[DEFAULT_POLICY_ID]
        policy.export_checkpoint(str(policies_dirpath), str(steps))
    print('Total experiment time in seconds: ', time.time() - start, flush=True)

# ========== main ========== #

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', help='experiment name')
    parser.add_argument('-m', '--max_steps', type=int, help='maximum number of steps', default=1e6)
    parser.add_argument('-p', '--port', help='ray redis port', default=None)
    parser.add_argument('-g', '--gpus', help='visible gpus', default='0')
    parser.add_argument('-f', '--freq', help='data frequency', default='1D')
    parser.add_argument('-e', '--env_cls', help='environment class', default=None)
    args = parser.parse_args(args)
    exp_name = args.exp_name
    max_steps = int(args.max_steps)
    port = args.port
    gpus = args.gpus
    freq = args.freq
    env_cls = args.env_cls
    if port:
        ray.init(address="127.0.0.1:{}".format(port))
    else:
        ray.init()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    exp_dirpath = Path('experiments').resolve() / exp_name
    rllib_env_config_filepath = exp_dirpath / 'rllib_env_config.yaml'
    rllib_trainer_config_filepath = exp_dirpath / 'rllib_trainer_config.yaml'
    rllib_env_config = yaml.load(rllib_env_config_filepath.open('r'), Loader=yaml.FullLoader)
    rllib_trainer_config = yaml.load(rllib_trainer_config_filepath.open('r'), Loader=yaml.FullLoader)
    rllib_trainer_cls = importlib.import_module('experiments.{}.custom_trainer'.format(exp_name)).CustomTrainer
    policies_dirpath = exp_dirpath / 'policies'
    if env_cls is not None:
        env_cls = importlib.import_module(env_cls).PortfolioEnv
    train(rllib_env_config, rllib_trainer_config, rllib_trainer_cls,
          policies_dirpath, max_steps=max_steps, env_cls=env_cls)

if __name__ == '__main__':
    main()
