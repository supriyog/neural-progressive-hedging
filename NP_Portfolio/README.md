This document explains the details of how to set up a conda environment for running the stochastic program (SP) and the proposed neural-progressive (NP) hedging approach for the paper titled "Augmenting Reinforcement Learning Policy Search with Stochastic Programming". In addition, we provide details of how to build and run other benchmark approaches, e.g., constrained RL (CPO, PPO-Lagrangian) and universal portfolio algorithms.

#####---------------------Basic Requirements (setting up CPLEX and Data)--------------------------------######
Download and install IBM ILOG CPLEX Studio 12.9 and set the python path variables in .bashrc file. Make sure 'import cplex' is working in python 3.7.
Download two datasets from this anonymous dropbox link: https://www.dropbox.com/sh/zuhz4tmxrgbkszy/AACAmNLJgGYVTcLlOXRsn6zna?dl=0
Copy both sandp500_2005_2019/ and prestored_samples/ folders within NP_Portfolio/data/ directory.

#####----------------Setting up environment for Stochastic program and neural-progressive algorithm------------------------######
Go to NP_Portfolio folder and run the following commands to set up a miniconda environment :
$ conda create --name=nphedging python=3.7
$ conda activate nphedging
$ source /Users/$USER/.bashrc
$ pip install -r pip_requirements.txt

#####----------------Customizing RLlib DDPG agent and policy network------------------------#####
Run command "$ which rllib" to find executable path for rllib. Let the output of command '$ which rllib' is 'Users/miniconda3/envs/nphedging/bin/rllib', then run the following bash script from NP_Portfolio folder with first argument as the outcome of '$ which rllib' except the last '/bin/rllib' part, i.e.,
$ sh rllib_replacements.sh Users/miniconda3/envs/nphedging

#####----------------Running SP and NP codes------------------------#####
A trained DDPG RL policy model is stored within NP_Portfolio/experiments/exp4004/ directory with checkpoints stored after every 3000 steps. For reproducing the results presented in the paper, policies at checkpoint 1500000 should be used. The data at prestored_samples/ directory (i.e., samples for Scenario tree, which has a depth of two and second layer has 1000 nodes, as indicated in config files) from anonymous dropbox link needs to be used for reproducing results. 
Set the system open file limit to 2048 as our codes use python multiprocessing, using this command: $ ulimit -n 2048
To run the NP codes with CVAR objective use following command:

$ python SPRL_expertDDPG_daily_CVAR.py --cf config/config_snp500_np.txt --rf results_NP --ef exp01 --en 0 --sn 0 --cv 0.99 --rld "exp4004" --cp 1500000

--cf: configuration file name in which stochastic programming related tunable parameters are mentioned. config/ folder includes two sample config file; config_snp500_np.txt has default settings for our neural-progressive approach, and config_snp500_sp.txt has required setting for pure stochastic programming. Therefore, to get results with pure SP, just provide config_snp500_sp.txt as config file.
--rf: result file name. The code will generate three results files (first with log, second one with returns, and third one stores weights or actions), whose names will start with the given --rf input value.
--ef: experiment folder name. A folder with this given name will be created within results/ folder (default value is "exp01").
--en: experiment number. Each run will generate results for consecutive 30 days. so, this integer value determines which sequence of 30 days to evaluate, i.e., a value of 0 (default) will generate resutls for first 30 days of 2019, a value of 1 generates results for day 31-60 and so on.
--sn: stock universe number. An integer value (default 0). data/stock_universes.csv file contains the list of stock universes in which each universe has 9 stocks. In experiments, we used first 4 universes (i.e., --sn value can vary from 0-3).
--cv: CVaR alpha value (takes a float value between 0 to 1) that determines the risk level. Default value is 0.99
--rld: DDPG RL generated policy directory. We stored a trained DDPG policy with checkpoints in experiments/exp4004/ directory. So, default value of --rld is 'exp4004'. If one needs to generate a new RL policy, then put the policy folder within experiments/ folder and mention the right name.
--cp: RL policy checkpoint value. Default value is 1500000. The code loads a DDPG policy with this checkpoint number from experiments/exp4004/policies/ folder. Note that, we stored checkpoints after every 3000 steps here, so the --cp value should be a multiplier of 3000.

For running NP with CVAR objective and liquidity constraint (which uses environment_constraint/ codes) run the following command:
$ python SPRL_expertDDPG_daily_Liquidity.py --cf config/config_snp500_np.txt --rf result_NPC --ef exp01 --en 0 --sn 0 --cv 0.99 --rld "exp4004" --cp 1500000

The code for running NP with liquidity constraints takes same set of inputs. However, note that this code uses environment_constraint/ directory to fetch information from environment which is different from the basic environment defined in environment/ folder.

*****Note: These runs generate 3 files within the results/'--ef value' folder. (1) A '.txt' file that starts with '--rf value', in which the entire log of NP or SP algorithms is stored. (2) A '.npy' file that starts with "Returns_+'--rf value'". This file stores a numpy array in which the returns in each time-step during evaluation is kept. One can use this file to plot our return results provided in the paper. (3) A '.npy' file that starts with "Weights_+'--rf value'". This file stores a 3-dimensional numpy array in which the weight allocation (i.e., actions) in each time-step during evaluation is kept. The neural-progressive iterative approach takes about 5-10 minutes (depending on the system preferences and number of CPUs to parrallelize) to converge to a decision in each time-step.

#####---------------- Evaluatign DDPG policies ------------------------#####
Run the following two codes to evaluate trained DDPG policies on portfolio environment. SPRL_eval_DDPG.py evaluates the basic DDPG policies and SPRL_eval_DDPG_Liquidity.py evaluates the DDPG policy after ensuring enough cash is reserved to ensure liquidity constraints (DDPG-H).
$ python SPRL_eval_DDPG.py --rf result_DDPG --ef exp01 --en 0 --sn 0 --rld "exp4004" --cp 1500000
$ python SPRL_eval_DDPG_Liquidity.py --rf result_DDPG-H --ef exp01 --en 0 --sn 0 --rld "exp4004" --cp 1500000

#####----------------Running and evaluatign Constraint RL benchmark algorithms (CPO and PPO-Lagrangian) codes------------------------#####
Set up mujoco, safety-gym and safety-starter-agents packages. Follow the instructions provided in "NP_Portfolio/Safety-RL-setup-guide.txt" file for setting it up in an Ubuntu machine. 
To train and evaluate CPO and PPO-Lagrangian approaches, go to NP_Portfolio/safety-starter-agents/scripts/ directory and run experiments.py (for training) and test_policy.py (for evaluation).

$ python experiments.py --algo 'cpo' --exp_name 'exp_cpo_v1' --seed 1001 --cpu 1 --sn 0
--alog : this takes the algorithm we want to run. Either provide 'cpo' or 'ppo_lagrangian'
--exp_name: provide any suitable string value. A directory with this name will be created within safety-starter-agents/data/ 
--seed: provide an integer value. a sub-directory within --exp_name valued folder will be created where all the trained policies will be stored.
--cpu: an integer value to represent how many CPUs to use for parallelization.

$ python test_policy.py ../data/exp_cpo_v1/exp_cpo_v1_s1001/ --episodes 1 --len 30 --en 0 --ef exp01 --rf results_cpo --sn 4 --itr 60000
In the first argument, we need to provide the location of the trainied policies. For our experiments.py example, it should be the given one.
--episodes: An integer value. Represents the number of episodes to evaluate. 
--len: An integer value. Represents the lengh of the episode (i.e., the number of steps to execute during evaluation).
--itr: An integer value. The checkpoint number from which the policy will be retrieved. By default, a checkpoint will be created after every 1000 episodes. So, this argument should be a multiplier of 1000.
--ef: The directory name (within NP_Portfolio/results/ directory) where results files will be stored. This (along with --en, --sn, --rf) is the same argument used in previous NP, SP examples. 

#####----------------Running and evaluatign Universal Portfolio benchmark algorithm codes------------------------#####
To run the UP algorithms, use the following code (It is better to create a separate conda environment to run these codes, as it needs python 3.6):
$ conda create --name=UP python=3.6
$ conda activate UP
$ pip install universal-portfolios PyYAML seaborn scipy==1.2.1 anytree bidict gym h5py

$ python Run_UnivPort_Algos.py --rf result_UP_v1 --ef exp01 --en 0 --sn 0 --an 0 --uf UPweights01

Here we are using two new command line inputs. --an is used for algorithm name ID (0 for OLMAR, 1 for PAMR, 2 for RMR). This run will generate allocation weights for that particular algorithm and for the given evaluation period. --uf takes the directory name where those generated weights numpy array files will be stored. It creates a folder within experiments/ folder. To evaluate these generated allocations in the environment, run the following code:

$ python Run_Evaluate_UP_algos_in_Environment.py --rf result_UP --ef exp01 --en 0 --sn 0 --an 0 --uf UPweights01

*****Note that we do not need to run UP algo for UCRP (as the weight allocation is known), but we can evaluate UCRP using Run_Evaluate_UP_algos_in_Environment.py; just provide the --an value as 3.

#####---------------- Training a DDPG policy from scratch ------------------------#####
For reproducing the results, there is no need to train a DDPG RL policy, as we have already included a trained policy within experiments/exp4004/ directory. However, if one needs to generate DDPG RL policies from stratch, then follow these steps.

In a new terminal, activate the nphedging environment (the one created for neural-progressive run), and run:
    python
    import ray
    ray.init() 

Note the ray port number from the new terminal and update the file 
	training/ray_default.py

From NP_Portfolio folder run the following python script (note that it will create 4 folder exp4001-exp4004 within experiments/ so rename our original trained policies within experiments/exp4004/ just to keep a backup) :
    $ python -m training.exp4001

The last step generates training files for all the experiments in the experiments/ directory. It creates 4 experiments with different train-test data split. We observe that exp4004 set up provides a better performance in general. Each experiment can be run by executing the associated bash script. For example, to start the DDPG RL training to generate our policies of experiments/exp4004/, run:
	$ ./training/exp4004.sh

This training bash code will generate policies within experiments/exp4004/policies/ directory and after every 3000 steps, a checkpoint will be created. One can then use these policies with a specific checkpoint to run the above mentioned neural-progressive algorithms.