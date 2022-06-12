
import random
import sys
import math
import time
import copy
import os
import re
import numpy as np
from datetime import datetime
import cplex
from cplex.exceptions import CplexError
from anytree import Node, RenderTree
from bidict import bidict
import multiprocessing
import threading
from multiprocessing import Manager
from multiprocessing.managers import BaseManager

sys.path.append("../")
from utils import utilities


class stocProgPH:

	#initialize the cplex model once from the master.
	def __init__(self, tree_file_name, eval_num_assets, config_file_name, current_allocation, result_file_name, alpha_cv):
		global slave_problem
		self.scen_tree_nodes, self.node_data = utilities.readJasonScenarioFile(tree_file_name)
		#self.cur_work_dir = cur_work_dir
		self.result_file = result_file_name
		self.param_dict = {}
		cfile = open(config_file_name)
		line = cfile.readline()
		while line != "":
			if "	#" in line:
				line = line.strip(" \r\n")
				line = line.split("	#")
				self.param_dict[line[1].split(" ")[0]] = line[0]
			line = cfile.readline()
		cfile.close()

		# for pre, fill, node in RenderTree(self.scen_tree_nodes[0]):
		# 	print("%s%s" % (pre, node.name))

		self.num_scenarios = len(self.scen_tree_nodes[0].leaves)
		self.num_nodes = len(self.scen_tree_nodes.keys())
		self.num_scen_per_subscen = int(self.param_dict["nscen_bundle"])  # min()
		#self.num_subscenarios = 3
		#self.scen_id_in_subscenarios = [[i*self.num_subscenarios+j for j in range(self.num_scen_per_subscen)] for i in range(self.num_subscenarios)]
		available_scenario_set = [i for i in range(self.num_scenarios)]
		self.scen_id_in_subscenarios = []
		new_scen_lists = []
		while len(available_scenario_set) > 0 :
			if(len(available_scenario_set) >= self.num_scen_per_subscen):
				if(self.param_dict["bundling_strategy"] == "random"):
					new_scen_lists = sorted(random.sample(available_scenario_set, self.num_scen_per_subscen))
				if (self.param_dict["bundling_strategy"] == "sequential"):
					new_scen_lists = sorted(available_scenario_set[0:self.num_scen_per_subscen])
			else:
				new_scen_lists = sorted(available_scenario_set)
			self.scen_id_in_subscenarios.append(new_scen_lists)
			available_scenario_set = list(set(available_scenario_set) - set(new_scen_lists))

		self.num_subscenarios = len(self.scen_id_in_subscenarios)
		# print (self.scen_id_in_subscenarios)
		print ("Number of scenarios and nodes in the Tree are: "+str(self.num_scenarios)+" & "+str(self.num_nodes))
		self.scenario_probabilities = [(1.0/(1.0*self.num_scenarios)) for i in range(self.num_scenarios)]#(1.0/(1.0*self.num_scenarios))
		self.node_sharing_scenarios = {}
		self.scenario_nodes = {}
		self.scenario_length = []

		scen_num = 0
		for scen in self.scen_tree_nodes[0].leaves:
			self.scenario_nodes[scen_num] = []
			for scen_node in range(len(scen.path)):
				self.scenario_nodes[scen_num].append(self.scen_tree_nodes.inverse[scen.path[scen_node]])
			self.scenario_length.append(len(self.scenario_nodes[scen_num]))
			scen_num += 1

		for scen_number in self.scenario_nodes.keys():
			for scen_node in self.scenario_nodes[scen_number]:
				if scen_node in self.node_sharing_scenarios.keys():
					self.node_sharing_scenarios[scen_node].append(scen_number)
				else:
					self.node_sharing_scenarios[scen_node] = [scen_number]

		self.node_level = [int(self.node_data[i]["time_step"]) for i in range(self.num_nodes)]

		self.horizon_length = max(self.scenario_length)
		self.num_assets = eval_num_assets   #int(self.param_dict["num_asset"]) #1#len(node_data[0]["interest_rate"])
		self.num_liabilities = int(self.param_dict["num_liability"]) #1

		self.interest_rates = [[[0.0 for ast in range(self.num_assets)] for time_step in range(self.horizon_length)] for scen in range(self.num_scenarios)]

		for nd in range(self.num_nodes):
			time_step = self.node_level[nd]
			all_interests_string = self.node_data[nd]["interest_rate"]
			all_interests_string = all_interests_string.strip(" \r\n")
			all_interests_string = all_interests_string.split(" ")
			for scen in self.node_sharing_scenarios[nd]:
				for asst in range(self.num_assets):
					self.interest_rates[scen][time_step][asst] = float(all_interests_string[asst])

		self.cash_interest_rates = [[self.node_data[self.scenario_nodes[scen][time_step]]["zc_price"] \
								for time_step in range(self.horizon_length)] for scen in range(self.num_scenarios)]			#check zc_price value and remove 0.01
		self.liabilitiy_amounts = [[[self.node_data[self.scenario_nodes[scen][time_step]]["liability"] for liab in range(self.num_liabilities)]\
								for time_step in range(self.horizon_length)] for scen in range(self.num_scenarios)]

		self.total_expected_liability = 0.0
		for scen in range(self.num_scenarios):
			for time_step in range(self.horizon_length):
				self.total_expected_liability = self.total_expected_liability + self.scenario_probabilities[scen]*sum(self.liabilitiy_amounts[scen][time_step])

		self.sibling_scenarios = [[[0 for k in range(self.horizon_length)] for j in range(len(self.scen_id_in_subscenarios[i]))] for i in range(self.num_subscenarios)]
		for sub_scen in range(self.num_subscenarios):
			sub_scen_nodes = {} #[self.scenario_nodes[i] for i in self.scen_id_in_subscenarios[sub_scen]]
			for i in range(len(self.scen_id_in_subscenarios[sub_scen])):
				sub_scen_nodes[i] = self.scenario_nodes[self.scen_id_in_subscenarios[sub_scen][i]]
			self.sibling_scenarios[sub_scen] = utilities.getSiblingScenarios(sub_scen_nodes, self.horizon_length, self.node_level, self.scen_tree_nodes)

		self.inverse_sibling_scenario = [[[0 for k in range(self.horizon_length)] for j in range(len(self.scen_id_in_subscenarios[i]))] for i in range(self.num_subscenarios)]
		for sub_scen in range(self.num_subscenarios):
			for scen in range(len(self.scen_id_in_subscenarios[sub_scen])):
				for time_step in range(self.horizon_length):
					self.inverse_sibling_scenario[sub_scen][self.sibling_scenarios[sub_scen][scen][time_step]][time_step] = scen

		self.initial_values = [current_allocation[i] for i in range(self.num_assets)]#initial_values
		self.initial_cash_value = current_allocation[-1]
		print (self.initial_values)
		print (self.initial_cash_value)

		self.total_initial_budget = sum(self.initial_values) + self.initial_cash_value
		#self.initial_cash_value = self.total_initial_budget - sum(self.initial_values)
		self.alpha_plus = float(self.param_dict["alpha_plus"]) # 0.01	#alpha_plus
		self.alpha_minus = float(self.param_dict["alpha_minus"]) #0.01	#alpha_minus
		self.parallel_option = str(self.param_dict["parallel_option"])

		self.force_convergence = self.param_dict["force_converge"]
		self.force_iteration = int(self.param_dict["force_iter"])
		self.learning_rate_nu = float(self.param_dict["lr_nu"])# 0.001	#tunable parameter
		self.learning_rate_lambda = float(self.param_dict["lr_lambda"])# 0.001
		self.alpha_neural_proximal_rate = float(self.param_dict["alpha_npr"])
		self.alpha_neural_rule = self.param_dict["dynamic_alpha_rule"]
		self.learning_alpha = 0.95
		self.learning_theta = 1.09
		self.learning_nu = 0.1
		self.learning_beta = 1.1
		self.learning_eta = 1.25
		self.learning_initiate_zeta = 0.5
		self.epsilon_convergence = 10**(-5)
		self.gamma_1 = 10**(-5)
		self.gamma_2 = 0.01
		self.gamma_3 = 0.25
		self.sigma = 10**(-5)
		self.global_count = 0
		self.max_iterations = 300# int(self.param_dict["max_iteration"])#1000
		self.alpha_cvar = alpha_cv #float(self.param_dict["alpha_cvar"])#0.75
		print ("CVAR value is: "+str(self.alpha_cvar))

		self.result_list = []
		self.slave_prob = [cplex.Cplex() for scen in range(self.num_subscenarios)]
		for scen in range(self.num_subscenarios):
			#self.slave_prob[scen] = cplex.Cplex()
			self.slave_prob[scen].objective.set_sense(self.slave_prob[scen].objective.sense.minimize)
			self.slave_prob[scen].set_log_stream(None)
			self.slave_prob[scen].set_error_stream(None)
			self.slave_prob[scen].set_warning_stream(None)
			self.slave_prob[scen].set_results_stream(None)
			self.slave_prob[scen].parameters.lpmethod.set(self.slave_prob[scen].parameters.lpmethod.values.network)
			self.slave_prob[scen].parameters.emphasis.memory.set(1)
			self.slave_prob[scen].parameters.mip.strategy.nodeselect.set(0)
			#self.slave_prob[scen].parameters.timelimit.set(600)
			#self.slave_prob[scen].parameters.mip.tolerances.mipgap.set(1)

	def log_result(self,result):
		# This is called whenever foo_pool(i) returns a result.
		# result_list is modified only by the main process, not the pool workers.
		self.result_list.append(result)

	def __call__(self, x):
		return self.solveStocProgSlaves(x)

	def solveStocProgMaster(self, weights_from_rl):
		fixed_learning_rate = 0.001
		rl_initial_allocation = [self.total_initial_budget*weights_from_rl[asst] for asst in range(self.num_assets+1)]
		rl_initial_buy = [max(0,(rl_initial_allocation[asst] - self.initial_values[asst])) for asst in range(self.num_assets)]
		rl_initial_sell = [max(0,(self.initial_values[asst]-rl_initial_allocation[asst])) for asst in range(self.num_assets)]
		stop_flag = 1
		lambda_dual_price = np.array([[[0.0 for ast in range(self.num_assets + 1)] for time_step in range(self.horizon_length)] for scen in range(self.num_scenarios)])
		average_portfolio_values = np.array([[[0.0 for ast in range(self.num_assets + 1)] for time_step in range(self.horizon_length)]  for scen in range(self.num_scenarios)])
		prev_average_values = np.array([[[0.0 for ast in range(self.num_assets + 1)] for time_step in range(self.horizon_length)]  for scen in range(self.num_scenarios)])
		prev_slave_xvalues = np.array([[[0.0 for ast in range(self.num_assets + 1)] for time_step in range(self.horizon_length)]  for scen in range(self.num_scenarios)])
		slave_current_xvalues = np.array([[[0.0 for ast in range(self.num_assets + 1)] for time_step in range(self.horizon_length)] for scen in range(self.num_scenarios)])
		slave_current_uvalues = np.array([[[[0.0 for ast in range(self.num_assets)] for time_step in range(self.horizon_length)] for scen in range(self.num_scenarios)] for upn in range(2)])
		prev_slave_sol = {}

		all_current_obj = np.array([0.0 for i in range(self.num_subscenarios)])
		current_Yvalues = np.array([0.0 for i in range(self.num_subscenarios)])
		prev_Yvalues = np.array([0.0 for i in range(self.num_subscenarios)])
		average_current_Yvalues = np.array([0.0 for i in range(self.num_subscenarios)])
		average_prev_Yvalues = np.array([0.0 for i in range(self.num_subscenarios)])
		average_Yvalues = 0.0
		lambda_y_dual = np.array([0.0 for i in range(self.num_subscenarios)])

		average_portfolio_node = np.array([[0.0 for j in range(self.num_assets + 1)] for i in range(self.num_nodes)])
		average_transaction_node = np.array([[[0.0 for k in range(2)]for j in range(self.num_assets)] for i in range(self.num_nodes)])
		processes = []
		pipe_list = []
		initial_time = time.time()

		while stop_flag :
			dual_average_total_value = 0.0
			dual_variable_total = 0.0
			#if (self.alpha_neural_rule == "fixed"):
			#	self.alpha_neural_proximal_rate = self.alpha_neural_proximal_rate
			if(self.alpha_neural_rule == "rt"):
				self.alpha_neural_proximal_rate = 1.0/math.sqrt(self.global_count+2)
			elif(self.alpha_neural_rule == "t"):
				self.alpha_neural_proximal_rate = 1.0/(self.global_count+2)
			elif(self.alpha_neural_rule == "t2"):
				self.alpha_neural_proximal_rate = 1.0/((self.global_count+2) ** 2)
				if(self.force_convergence == "T"):
					if(self.global_count >= self.force_iteration):
						self.alpha_neural_proximal_rate = 0.0
			elif(self.alpha_neural_rule == "t3"):
				self.alpha_neural_proximal_rate = 1.0/((self.global_count+2) ** 3)

			if(self.global_count == 0):
				pName = ""
				send_end = ""
				for scen_number in range(self.num_subscenarios):
					num_scen_in_scen_number = len(self.scen_id_in_subscenarios[scen_number])
					scen_id = [self.scen_id_in_subscenarios[scen_number][scen_count] for scen_count in range(num_scen_in_scen_number)]
					scenario_results, obj = self.solveStocProgSlaves(pName, scen_number, lambda_dual_price, lambda_y_dual, average_portfolio_values, average_Yvalues, [],[])
					prev_slave_sol[scen_number] = np.array(scenario_results)
					dual_average_total_value += obj
					all_current_obj[scen_number] = obj
					dual_variable_total += scenario_results[-2]
					for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
						dual_variable_total += (1.0*scenario_results[-3-scen_count])/((1-self.alpha_cvar)*self.num_scenarios)#len(self.scen_id_in_subscenarios[scen_number])
					#print (obj, scenario_results[-5:])
					current_Yvalues[scen_number] = scenario_results[-2]
					for scen_count in range(num_scen_in_scen_number):
						for time_step in range(self.horizon_length):
							for asset_num in range(self.num_assets):
								slave_current_xvalues[scen_id[scen_count]][time_step][asset_num] = scenario_results[scen_count*self.horizon_length*self.num_assets+\
								                                                                        time_step*self.num_assets + asset_num]
								slave_current_uvalues[0][scen_id[scen_count]][time_step][asset_num] = scenario_results[self.horizon_length*num_scen_in_scen_number* self.num_assets +\
								                                                                        scen_count*self.horizon_length*self.num_assets+time_step * self.num_assets + asset_num]
								slave_current_uvalues[1][scen_id[scen_count]][time_step][asset_num] = scenario_results[2 *num_scen_in_scen_number* self.horizon_length * self.num_assets +\
								                                                                               scen_count*self.horizon_length*self.num_assets+time_step * self.num_assets + asset_num]
							slave_current_xvalues[scen_id[scen_count]][time_step][self.num_assets] = scenario_results[3*num_scen_in_scen_number*self.horizon_length*self.num_assets + scen_count*self.horizon_length+time_step]
				#exit()

			else:
				if(self.parallel_option == "mthread"):
					#pool = multiprocessing.Pool(processes = 8)
					pool = multiprocessing.pool.ThreadPool(processes = 8)
					for scen_number in range(self.num_subscenarios):
						pool.apply_async(self.solveStocProgSlaves, args = (self.parallel_option, scen_number, lambda_dual_price, lambda_y_dual, average_portfolio_values, average_Yvalues, prev_slave_sol[scen_number], []), callback = self.log_result) #, callback = log_result)
						#print(res.get(timeout = 1000))
					pool.close()
					pool.join()

				if(self.parallel_option == "mprocess"):
					#results = []
					for scen_number in range(self.num_subscenarios):
						process_name = str(scen_number) #self.global_count * self.num_scenarios +
						recv_end, send_end = multiprocessing.Pipe(False)
						new_process = multiprocessing.Process(target = self.solveStocProgSlaves, args = (self.parallel_option, scen_number, lambda_dual_price, lambda_y_dual, average_portfolio_values, average_Yvalues, prev_slave_sol[scen_number],send_end))
						processes.append(new_process)
						pipe_list.append(recv_end)
						new_process.start()

					for proc in processes:
						proc.join()
					for proc in processes:
						proc.terminate()
					del processes[:]
					processes = []
					self.result_list = [x.recv() for x in pipe_list]

				prev_slave_sol = {}
				for rn in range(len(self.result_list)):
					subProcess = self.result_list[rn][0]
					scenario_results = self.result_list[rn][1:]
					#print (scenario_results[-5:])
					scen_number = subProcess
					all_current_obj[scen_number] = scenario_results[-1]
					prev_slave_sol[scen_number] = scenario_results[0:-1]
					num_scen_in_scen_number = len(self.scen_id_in_subscenarios[scen_number])
					scen_id = [self.scen_id_in_subscenarios[scen_number][scen_count] for scen_count in range(num_scen_in_scen_number)]
					dual_average_total_value += scenario_results[-1]
					dual_variable_total += scenario_results[-3]
					for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
						dual_variable_total += (1.0*scenario_results[-4-scen_count])/((1-self.alpha_cvar)*self.num_scenarios)#len(self.scen_id_in_subscenarios[scen_number])
					current_Yvalues[scen_number] = scenario_results[-3]
					for scen_count in range(num_scen_in_scen_number):
						for time_step in range(self.horizon_length):
							for asset_num in range(self.num_assets):
								slave_current_xvalues[scen_id[scen_count]][time_step][asset_num] = scenario_results[scen_count * self.horizon_length * self.num_assets + \
																									time_step * self.num_assets + asset_num]
								slave_current_uvalues[0][scen_id[scen_count]][time_step][asset_num] = scenario_results[self.horizon_length * num_scen_in_scen_number * self.num_assets + \
																								scen_count * self.horizon_length * self.num_assets + time_step * self.num_assets + asset_num]
								slave_current_uvalues[1][scen_id[scen_count]][time_step][asset_num] = scenario_results[2 * num_scen_in_scen_number * self.horizon_length * self.num_assets + \
																								scen_count * self.horizon_length * self.num_assets + time_step * self.num_assets + asset_num]
							slave_current_xvalues[scen_id[scen_count]][time_step][self.num_assets] = scenario_results[3 * num_scen_in_scen_number * self.horizon_length * self.num_assets + scen_count * self.horizon_length + time_step]

				#pipe_list = []
				self.result_list = []
				pipe_list = []


			for nd in range(self.num_nodes):
				time_step = self.node_level[nd]
				total_value_current_asset = [0.0 for asset_num in range(self.num_assets+1)]
				total_buy_asset = [0.0 for asset_num in range(self.num_assets+1)]
				total_sell_asset = [0.0 for asset_num in range(self.num_assets+1)]
				sum_scen_probability = 0.0
				for scen in self.node_sharing_scenarios[nd]:
					sum_scen_probability += self.scenario_probabilities[scen]
					for asset_num in range(self.num_assets+1):
						total_value_current_asset[asset_num] += slave_current_xvalues[scen][time_step][asset_num]#self.scenario_probabilities[scen]*
						if(asset_num < self.num_assets):
							total_buy_asset[asset_num] += slave_current_uvalues[0][scen][time_step][asset_num]
							total_sell_asset[asset_num] += slave_current_uvalues[1][scen][time_step][asset_num]

				for asset_num in range(self.num_assets + 1):
					average_portfolio_node[nd][asset_num] = total_value_current_asset[asset_num] / (1.0*len(self.node_sharing_scenarios[nd]))#sum_scen_probability
					if (asset_num < self.num_assets):
						average_transaction_node[nd][asset_num][0] = total_buy_asset[asset_num] / (1.0*len(self.node_sharing_scenarios[nd]))#sum_scen_probability
						average_transaction_node[nd][asset_num][1] = total_sell_asset[asset_num] / (1.0*len(self.node_sharing_scenarios[nd]))#sum_scen_probability

				for scen in self.node_sharing_scenarios[nd]:
					for asset_num in range(self.num_assets+1):
						average_portfolio_values[scen][time_step][asset_num] = total_value_current_asset[asset_num] / (1.0*len(self.node_sharing_scenarios[nd]))

			average_Yvalues = sum(current_Yvalues) / (1.0 * len(current_Yvalues))
			for scen in range(self.num_subscenarios):
				average_current_Yvalues[scen] = average_Yvalues

			root_node_val = 0
			time_step = 0#self.node_level[root_node_val]
			for scen in self.node_sharing_scenarios[root_node_val]:
				for asset_num in range(self.num_assets+1):
					average_portfolio_values[scen][time_step][asset_num] = self.alpha_neural_proximal_rate*rl_initial_allocation[asset_num] + (1-self.alpha_neural_proximal_rate)*average_portfolio_values[scen][time_step][asset_num]

			for asset_num in range(self.num_assets + 1):
				average_portfolio_node[0][asset_num] = self.alpha_neural_proximal_rate*rl_initial_allocation[asset_num] + (1-self.alpha_neural_proximal_rate)*average_portfolio_node[0][asset_num]
				if (asset_num < self.num_assets):
					average_transaction_node[0][asset_num][0] = self.alpha_neural_proximal_rate*rl_initial_buy[asset_num] + (1-self.alpha_neural_proximal_rate)*average_transaction_node[0][asset_num][0]
					average_transaction_node[0][asset_num][1] = self.alpha_neural_proximal_rate*rl_initial_sell[asset_num] + (1-self.alpha_neural_proximal_rate)*average_transaction_node[0][asset_num][1]


			total_residual_value = 0.0
			residual_denominator = 0.0
			for scen_number in range(self.num_scenarios):
				for time_step in range(self.horizon_length):
					for asset_num in range(self.num_assets+1):
						total_residual_value += self.scenario_probabilities[scen_number] * (( slave_current_xvalues[scen_number][time_step][asset_num] - prev_average_values[scen_number][time_step][asset_num]) ** 2)
						residual_denominator += self.scenario_probabilities[scen_number] * ((prev_average_values[scen_number][time_step][asset_num]) ** 2)
			#-------- Check later, convergence computation, should we use some scenario probability into it? ----#
			for scen_number in range(self.num_subscenarios):
				total_residual_value += ((current_Yvalues[scen_number] - average_prev_Yvalues[scen_number]) ** 2)/(1.0*self.num_subscenarios)
				residual_denominator += (average_prev_Yvalues[scen_number] ** 2)/(1.0*self.num_subscenarios)

			if(residual_denominator > 0):
				total_residual_value = math.sqrt(total_residual_value/(1.0*max(1,residual_denominator)))
			else:
				total_residual_value = 10**5

			######## initialize the learning parameter
			if(self.global_count == 0):
				print (dual_average_total_value)

				numerator_value = max(1, abs(2*self.learning_initiate_zeta*dual_average_total_value))
				denominator_value = max(1, np.linalg.norm(average_portfolio_values - slave_current_xvalues)**2 + np.linalg.norm(current_Yvalues - average_current_Yvalues)**2)
				self.learning_rate_lambda = numerator_value / (1.0*denominator_value)
				print ("Initial learning rate is: "+str(self.learning_rate_lambda))

			#### Update lambda dual variables in master
			for scen_number in range(self.num_scenarios):
				for time_step in range(self.horizon_length):
					for asset_num in range(self.num_assets+1):
						lambda_dual_price[scen_number][time_step][asset_num] = lambda_dual_price[scen_number][time_step][asset_num] + self.learning_rate_lambda*(slave_current_xvalues[scen_number][time_step][asset_num] - \
																				average_portfolio_values[scen_number][time_step][asset_num])

			for scen_number in range(self.num_subscenarios):
				lambda_y_dual[scen_number] = lambda_y_dual[scen_number] + self.learning_rate_lambda * (current_Yvalues[scen_number] - average_current_Yvalues[scen_number])

			######## Update learning parameter according to Zehtabian et. al. (CIRRELT) [:,:reduced_horizon,:]
			reduced_horizon = self.horizon_length - 1
			if(self.global_count > 0):
				change_in_average = ((np.linalg.norm(average_portfolio_values - prev_average_values)**2)+(np.linalg.norm(average_current_Yvalues - average_prev_Yvalues)**2))/ \
				                    (1.0*max(np.linalg.norm(average_portfolio_values)**2 + np.linalg.norm(average_current_Yvalues)**2, np.linalg.norm(prev_average_values)**2 + np.linalg.norm(average_prev_Yvalues)**2))
				proximal_change = self.learning_rate_lambda*(np.linalg.norm(slave_current_xvalues - average_portfolio_values) ** 2 + np.linalg.norm(current_Yvalues - average_current_Yvalues)**2)
				lagrangian_change = 0.0#self.sigma*(dual_average_total_value - 0.5*self.learning_rate_lambda*np.linalg.norm(slave_current_xvalues - prev_average_values))
				for scen_number in range(self.num_scenarios):
					for time_step in range(self.horizon_length-1):
						for asset_num in range(self.num_assets + 1):
							lagrange_penalty_cur = (lambda_dual_price[scen_number][time_step][asset_num]*(slave_current_xvalues[scen_number][time_step][asset_num]-prev_average_values[scen_number][time_step][asset_num]))
							# if(time_step == self.horizon_length-1):
							# 	lagrangian_change += lagrange_penalty_cur - self.scenario_probabilities[scen_number]*slave_current_xvalues[scen_number][time_step][asset_num]
							# else:
							lagrangian_change += lagrange_penalty_cur       # Update this based on the objective function

				for scen_number in range(self.num_subscenarios):
					lagrangian_change += lambda_y_dual[scen_number]*(current_Yvalues[scen_number]-average_prev_Yvalues[scen_number])
				lagrangian_change += dual_variable_total

				if((change_in_average >= self.gamma_1) or (proximal_change >= self.sigma*abs(lagrangian_change))):
					condition_1 = ((np.linalg.norm(average_portfolio_values - prev_average_values)**2 + np.linalg.norm(average_current_Yvalues - average_prev_Yvalues)**2) - \
					               (np.linalg.norm(slave_current_xvalues - average_portfolio_values)**2 + np.linalg.norm(current_Yvalues - average_current_Yvalues)**2)) / \
					              (1.0*max(1,(np.linalg.norm(slave_current_xvalues - average_portfolio_values)**2 +np.linalg.norm(current_Yvalues - average_current_Yvalues)**2)))
					condition_2 = ((np.linalg.norm(slave_current_xvalues - average_portfolio_values)**2 + np.linalg.norm(current_Yvalues - average_current_Yvalues)**2) - \
					               (np.linalg.norm(average_portfolio_values - prev_average_values)**2 + np.linalg.norm(average_current_Yvalues - average_prev_Yvalues)**2))/ \
					              (1.0*max(1,(np.linalg.norm(average_portfolio_values - prev_average_values)**2 + np.linalg.norm(average_current_Yvalues - average_prev_Yvalues)**2)))
					if(condition_1 > self.gamma_2):
						self.learning_rate_lambda = self.learning_rate_lambda * self.learning_alpha
					elif(condition_2 > self.gamma_3):
						self.learning_rate_lambda = self.learning_rate_lambda * self.learning_theta
					else:
						self.learning_rate_lambda = self.learning_rate_lambda
				elif((np.linalg.norm(slave_current_xvalues - average_portfolio_values)**2 + np.linalg.norm(current_Yvalues - average_current_Yvalues)**2) > (np.linalg.norm(prev_slave_xvalues - prev_average_values)**2 + np.linalg.norm(prev_Yvalues - average_prev_Yvalues)**2)):
					if((((np.linalg.norm(slave_current_xvalues - average_portfolio_values)**2 + np.linalg.norm(current_Yvalues - average_current_Yvalues)**2) - (np.linalg.norm(prev_slave_xvalues - prev_average_values)**2 + np.linalg.norm(prev_Yvalues - average_prev_Yvalues)**2)) / \
					    (1.0*(np.linalg.norm(prev_slave_xvalues - prev_average_values)**2 + np.linalg.norm(prev_Yvalues - average_prev_Yvalues)**2))) > self.learning_nu):
						self.learning_rate_lambda = self.learning_rate_lambda * self.learning_beta
					else:
						self.learning_rate_lambda = self.learning_rate_lambda
				else:
					self.learning_rate_lambda = self.learning_rate_lambda * self.learning_eta

				#self.learning_rate_lambda =  0.001
			#############################################################################################################################

			for scen_number in range(self.num_scenarios):
				for time_step in range(self.horizon_length):
					for asset_num in range(self.num_assets+1):
						prev_average_values[scen_number][time_step][asset_num] = average_portfolio_values[scen_number][time_step][asset_num]
						prev_slave_xvalues[scen_number][time_step][asset_num] = slave_current_xvalues[scen_number][time_step][asset_num]

			for scen_number in range(self.num_subscenarios):
				prev_Yvalues[scen_number] = current_Yvalues[scen_number]
				average_prev_Yvalues[scen_number] = average_current_Yvalues[scen_number]

			utilities.write_results("total residual value, dual objective, learning rate, time, np-alpha at iteration " +str(self.global_count) +" ### " + str(total_residual_value) + " ### "+str(dual_average_total_value) + " ### "+str(self.learning_rate_lambda)+" ### "+str(time.time()-initial_time)+" ### "+str(self.alpha_neural_proximal_rate), self.result_file)
			if(self.global_count % 20 == 1):
				current_average_sol = []
				for nd in range(self.num_nodes - self.num_scenarios):
					current_average_sol.append(average_portfolio_node[nd])

			if(self.global_count >= self.max_iterations):
				utilities.write_results("total residual value at convergence is " +str(self.global_count) +" ### " + str(total_residual_value) +" ### " + str(self.learning_rate_lambda), self.result_file)
				stop_flag = 0
			if(total_residual_value <= self.epsilon_convergence):
				current_average_sol = []
				for nd in range(self.num_nodes - self.num_scenarios):
					current_average_sol.append(average_portfolio_node[nd])
				utilities.write_results("total residual value at convergence is " +str(self.global_count) +" ### " + str(total_residual_value) +" ### " + str(self.learning_rate_lambda), self.result_file)
				stop_flag = 0

			self.global_count += 1

		utilities.write_results("Total time taken is: "+str(time.time()-initial_time),self.result_file)
		return average_portfolio_node[0],average_transaction_node[0]

	def solveStocProgSlaves(self, parallel_option, scen_number, lambda_values, lambda_y, average_Xvalues, average_Yvalue, warm_start_sol, send_end):
		my_sense1 = ""
		row1 = [0.0 for k in range(1)]
		my_rownames1 = ["" for k in range(1)]
		big_M = 9999
		ubs = [0.0 for k in range(1)]
		lbs = [0.0 for k in range(1)]
		objc = [0.0 for k in range(1)]
		my_colnames = ["" for k in range(1)]
		raw1 = [[[0 for k in range(1)] for j in range(2)] for l in range(1)]
		typ1 = ["" for k in range(1)]
		num_scen_in_scen_number = len(self.scen_id_in_subscenarios[scen_number])
		scen_id = [self.scen_id_in_subscenarios[scen_number][scen_count] for scen_count in range(len(self.scen_id_in_subscenarios[scen_number]))]

		if self.global_count == 0:
			############ Add X variables ##########################
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length):
					for asset_num in range(self.num_assets):
						objc[0] = 0.0
						ubs[0] = math.inf			# upper bound for the variable
						lbs[0] = 0.0						# lower bound for the variable
						my_colnames[0] = "X-" + str(scen_count) + "-" + str(time_step) + "-" + str(asset_num)	# name/identifier of the variable
						raw1[0][0] = []
						raw1[0][1] = []
						self.slave_prob[scen_number].variables.add(obj = objc, ub = ubs, lb = lbs, names = my_colnames, columns=raw1)

			############ Add u+ variables ##########################
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length):
					for asset_num in range(self.num_assets):
						objc[0] = 0.0			#Coefficient of the variable in objective
						ubs[0] = math.inf			# upper bound for the variable
						lbs[0] = 0						# lower bound for the variable
						my_colnames[0] = "UP-" + str(scen_count) + "-" + str(time_step) + "-" + str(asset_num)	# name/identifier of the variable
						raw1[0][0] = []
						raw1[0][1] = []
						self.slave_prob[scen_number].variables.add(obj = objc, ub = ubs, lb = lbs, names = my_colnames, columns=raw1)

			############ Add u- variables ##########################
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length):
					for asset_num in range(self.num_assets):
						objc[0] = 0.0			#Coefficient of the variable in objective
						ubs[0] = math.inf			# upper bound for the variable
						lbs[0] = 0						# lower bound for the variable
						my_colnames[0] = "UN-" + str(scen_count) + "-" + str(time_step) + "-" + str(asset_num)	# name/identifier of the variable
						raw1[0][0] = []
						raw1[0][1] = []
						self.slave_prob[scen_number].variables.add(obj = objc, ub = ubs, lb = lbs, names = my_colnames, columns=raw1)

			############ Add Liquid variables L ##########################
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length):
					objc[0] = 0.0
					ubs[0] = math.inf			# upper bound for the variable
					lbs[0] = 0						# lower bound for the variable
					#typ1 = "I"						# type for the variable
					my_colnames[0] = "LQ-" + str(scen_count) + "-" + str(time_step)	# name/identifier of the variable
					raw1[0][0] = []
					raw1[0][1] = []
					self.slave_prob[scen_number].variables.add(obj = objc, ub = ubs, lb = lbs, names = my_colnames, columns=raw1)

			############ Add proximal term p ##########################
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length):
					for asset_num in range(self.num_assets+1):
						objc[0] = 0.0			#Coefficient of the variable in objective
						ubs[0] = math.inf			# upper bound for the variable
						lbs[0] = -1.0*math.inf		# lower bound for the variable
						my_colnames[0] = "P-" + str(scen_count) + "-" + str(time_step) + "-" + str(asset_num)	# name/identifier of the variable
						raw1[0][0] = []
						raw1[0][1] = []
						self.slave_prob[scen_number].variables.add(obj = objc, ub = ubs, lb = lbs, names = my_colnames, columns=raw1)

			#-------- Add the a(s) variable to learize CVAR equation max(0, f(x)-y) ----------------#
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				objc[0] = (1.0*self.scenario_probabilities[self.scen_id_in_subscenarios[scen_number][scen_count]])/(1-self.alpha_cvar)##len(self.scen_id_in_subscenarios[scen_number]))			#Coefficient of the variable in objective
				ubs[0] = math.inf			# upper bound for the variable
				lbs[0] = 0.0	# lower bound for the variable
				my_colnames[0] = "a-" + str(scen_count)
				raw1[0][0] = []
				raw1[0][1] = []
				self.slave_prob[scen_number].variables.add(obj = objc, ub = ubs, names = my_colnames, columns=raw1)


			#-------- Add the y(s) variable for CVAR computation ----------------#
			objc[0] = 1.0  # Coefficient of the variable in objective
			ubs[0] = math.inf  # upper bound for the variable
			lbs[0] = -10.0  # lower bound for the variable
			my_colnames[0] = "y"
			raw1[0][0] = []
			raw1[0][1] = []
			self.slave_prob[scen_number].variables.add(obj = objc, ub = ubs, lb=lbs, names = my_colnames, columns = raw1)

			#-------- Add the y(s)-y variable for CVAR computation ----------------#
			objc[0] = 0.0  # Coefficient of the variable in objective
			ubs[0] = math.inf  # upper bound for the variable
			lbs[0] = -1.0*math.inf   # lower bound for the variable
			my_colnames[0] = "yres"
			raw1[0][0] = []
			raw1[0][1] = []
			self.slave_prob[scen_number].variables.add(obj = objc, ub = ubs, lb = lbs, names = my_colnames, columns = raw1)

			starting_UP = num_scen_in_scen_number*self.horizon_length*self.num_assets
			starting_UN = 2*num_scen_in_scen_number*self.horizon_length*self.num_assets
			starting_LQ = 3*num_scen_in_scen_number*self.horizon_length*self.num_assets
			starting_P = self.horizon_length*num_scen_in_scen_number*(3*self.num_assets+1)
			starting_a = self.horizon_length*num_scen_in_scen_number*(4*self.num_assets+2)
			starting_b = (self.horizon_length*num_scen_in_scen_number*(4*self.num_assets+2))+ num_scen_in_scen_number
			starting_y = (self.horizon_length*num_scen_in_scen_number*(4*self.num_assets+2))+num_scen_in_scen_number    #2*

			#### constraint: r_t(i)x_t(i)+(1-alpha^+)u^+_t(i)-(1+alpha^-)u^-_t(i) = x_{t+1}(i) \forall i\in Assets\0, t\in Time ###
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length):
					for asset_num in range(self.num_assets):
						my_sense1 = "E"			# Sense of constraint, E, L, or G.
						row1[0] = 0.0			# r.h.s. constant of the constraint #time_step*self.num_assets + asset_num
						if(time_step == 0):
							row1[0] = (-1.0)*self.initial_values[asset_num]
						my_rownames1[0]="c01-" + str(scen_count) + "-" + str(time_step) + "-" + str(asset_num)	# Name/identifier of constraints
						raw1[0][0] = []
						raw1[0][1] = []

						raw1[0][0].append(scen_count*self.horizon_length*self.num_assets+time_step*self.num_assets + asset_num)
						raw1[0][1].append(-1.0)

						raw1[0][0].append(starting_UP + scen_count*self.horizon_length*self.num_assets+time_step*self.num_assets + asset_num)
						raw1[0][1].append(1.0 - self.alpha_plus)

						raw1[0][0].append(starting_UN + scen_count*self.horizon_length*self.num_assets+ time_step*self.num_assets + asset_num)
						raw1[0][1].append(-1.0 - self.alpha_minus)

						if(time_step > 0):#< self.horizon_length-1):
							raw1[0][0].append(scen_count*self.horizon_length*self.num_assets+(time_step-1) * self.num_assets + asset_num)
							raw1[0][1].append(1+self.interest_rates[scen_id[scen_count]][time_step][asset_num])	#check this interest rate info later (1+r) or just r

						self.slave_prob[scen_number].linear_constraints.add(rhs = row1, senses = my_sense1, lin_expr= raw1, names = my_rownames1)

			#### Add measurable constraints (Currently this is taken from stocProgPH.py)###
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length):
					#print(self.sibling_scenarios[scen_number][scen_count][time_step])
					for asset_num in range(self.num_assets + 1):
						my_sense1 = "E"  # Sense of constraint, E, L, or G.
						row1[0] = 0.0  # r.h.s. constant of the constraint #time_step*self.num_assets + asset_num
						my_rownames1[0] = "c02-" + str(scen_count) + "-" + str(time_step) + "-" + str(asset_num)  # Name/identifier of constraints
						raw1[0][0] = []
						raw1[0][1] = []

						if (asset_num == self.num_assets):
							raw1[0][0].append(starting_LQ + scen_count * self.horizon_length + time_step)
							raw1[0][1].append(1.0)
							raw1[0][0].append(starting_LQ + self.sibling_scenarios[scen_number][scen_count][time_step] * self.horizon_length + time_step)
							raw1[0][1].append(-1.0)
						else:
							raw1[0][0].append(scen_count * self.horizon_length * self.num_assets + time_step * self.num_assets + asset_num)
							raw1[0][1].append(1.0)
							raw1[0][0].append(self.sibling_scenarios[scen_number][scen_count][time_step] * self.horizon_length * self.num_assets + time_step * self.num_assets + asset_num)
							raw1[0][1].append(-1.0)

						if (scen_count != self.sibling_scenarios[scen_number][scen_count][time_step]):
							self.slave_prob[scen_number].linear_constraints.add(rhs = row1, senses = my_sense1, lin_expr = raw1, names = my_rownames1)

			#### Add simple borrowing constraints with positive liquidity###
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length):
					my_sense1 = "E"			# Sense of constraint, E, L, or G.
					row1[0] = sum(self.liabilitiy_amounts[scen_id[scen_count]][time_step]) # r.h.s. constant of the constraint #time_step*self.num_assets + asset_num
					if (time_step == 0):
						row1[0] = (-1.0) * self.initial_cash_value + sum(self.liabilitiy_amounts[scen_id[scen_count]][time_step])
					my_rownames1[0]="c03-" + str(scen_count) + "-" + str(time_step)	# Name/identifier of constraints
					raw1[0][0] = []
					raw1[0][1] = []

					raw1[0][0].append(starting_LQ + scen_count*self.horizon_length+time_step)
					raw1[0][1].append(-1.0)

					for asset_num in range(self.num_assets):
						raw1[0][0].append(starting_UP + scen_count*self.horizon_length*self.num_assets+ time_step * self.num_assets + asset_num)
						raw1[0][1].append(-1.0)

						raw1[0][0].append(starting_UN + scen_count*self.horizon_length*self.num_assets+ time_step * self.num_assets + asset_num)
						raw1[0][1].append(1.0)

					if (time_step > 0):  # < self.horizon_length-1):
						raw1[0][0].append(starting_LQ + scen_count*self.horizon_length+ time_step-1)
						raw1[0][1].append(1+self.cash_interest_rates[scen_id[scen_count]][time_step])  # check this interest rate info later

					self.slave_prob[scen_number].linear_constraints.add(rhs = row1, senses = my_sense1, lin_expr= raw1, names = my_rownames1)

			#### priximal constraint p=x-x^k ###
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length):
					for asset_num in range(self.num_assets+1):
						my_sense1 = "E"			# Sense of constraint, E, L, or G.
						row1[0] = 0.0			# r.h.s. constant of the constraint #time_step*self.num_assets + asset_num
						my_rownames1[0]="c04-" + str(scen_count) + "-" + str(time_step) + "-" + str(asset_num)	# Name/identifier of constraints
						raw1[0][0] = []
						raw1[0][1] = []

						if(asset_num == self.num_assets):
							raw1[0][0].append(starting_LQ + scen_count*self.horizon_length + time_step)
							raw1[0][1].append(-1.0)
						else:
							raw1[0][0].append(scen_count*self.horizon_length*self.num_assets + time_step*self.num_assets + asset_num)
							raw1[0][1].append(-1.0)

						raw1[0][0].append(starting_P + scen_count*(self.horizon_length)*(self.num_assets+1) + time_step*(self.num_assets+1) + asset_num)
						raw1[0][1].append(1.0)

						self.slave_prob[scen_number].linear_constraints.add(rhs = row1, senses = my_sense1, lin_expr= raw1, names = my_rownames1)


			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				my_sense1 = "G"  # Sense of constraint, E, L, or G.
				row1[0] = self.total_initial_budget#0.0  # r.h.s. constant of the constraint #time_step*self.num_assets + asset_num
				my_rownames1[0] = "c05-" + str(scen_count)
				raw1[0][0] = []
				raw1[0][1] = []
				for asset_num in range(self.num_assets):
					raw1[0][0].append(scen_count * self.horizon_length * self.num_assets + (self.horizon_length-1) * self.num_assets + asset_num)
					raw1[0][1].append(1.0)
				raw1[0][0].append(starting_LQ + scen_count*self.horizon_length + self.horizon_length -1)
				raw1[0][1].append(1.0)
				raw1[0][0].append(starting_a + scen_count)
				raw1[0][1].append(1.0)
				raw1[0][0].append(starting_y)
				raw1[0][1].append(1.0)
				self.slave_prob[scen_number].linear_constraints.add(rhs = row1, senses = my_sense1, lin_expr = raw1, names = my_rownames1)


			my_sense1 = "E"  # Sense of constraint, E, L, or G.
			row1[0] = 0.0  # r.h.s. constant of the constraint #time_step*self.num_assets + asset_num
			my_rownames1[0] = "c083"
			raw1[0][0] = []
			raw1[0][1] = []
			raw1[0][0].append(starting_y)
			raw1[0][1].append(1.0)
			raw1[0][0].append(starting_y+1)
			raw1[0][1].append(-1.0)
			self.slave_prob[scen_number].linear_constraints.add(rhs = row1, senses = my_sense1, lin_expr = raw1, names = my_rownames1)

		else:
			starting_P = self.horizon_length*num_scen_in_scen_number*(3*self.num_assets+1)
			starting_LQ = 3 * num_scen_in_scen_number * self.horizon_length * self.num_assets
			starting_a = self.horizon_length*num_scen_in_scen_number*(4*self.num_assets+2)
			starting_y = (self.horizon_length*num_scen_in_scen_number*(4*self.num_assets+2))+num_scen_in_scen_number

			#---------- add proximal and lagrangian term for x variables ------------#
			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length-1):
					for asset_num in range(self.num_assets+1):
						self.slave_prob[scen_number].objective.set_quadratic_coefficients(starting_P+scen_count*(self.horizon_length)*(self.num_assets+1)+time_step*(self.num_assets+1) + asset_num,\
														starting_P+scen_count*(self.horizon_length)*(self.num_assets+1)+time_step*(self.num_assets+1) + asset_num, (0.5)*self.learning_rate_lambda)

			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length-1):
					for asset_num in range(self.num_assets+1):
						base_updated_coefficient = (1.0) * lambda_values[scen_id[scen_count]][time_step][asset_num]  # - \ lambda_values[self.inverse_sibling_scenario[scen_number][time_step]][time_step][asset_num])
						self.slave_prob[scen_number].objective.set_linear(starting_P+scen_count*(self.horizon_length)*(self.num_assets+1)+time_step*(self.num_assets+1) + asset_num, base_updated_coefficient)

			for scen_count in range(len(self.scen_id_in_subscenarios[scen_number])):
				for time_step in range(self.horizon_length):
					for asset_num in range(self.num_assets+1):
						self.slave_prob[scen_number].linear_constraints.set_rhs("c04-" + str(scen_count) + "-" + str(time_step) + "-" + str(asset_num), -1.0*average_Xvalues[scen_id[scen_count]][time_step][asset_num])

			# ---------- add proximal and lagrangian term for y variables ------------#
			self.slave_prob[scen_number].objective.set_quadratic_coefficients(starting_y + 1, starting_y + 1,(0.5) * self.learning_rate_lambda)
			self.slave_prob[scen_number].objective.set_linear(starting_y + 1 , lambda_y[scen_number])
			self.slave_prob[scen_number].linear_constraints.set_rhs("c083", average_Yvalue)
			self.slave_prob[scen_number].variables.set_upper_bounds(starting_y , math.inf)

			self.slave_prob[scen_number].start.set_start(col_status=[],row_status=[],col_primal=warm_start_sol,row_primal=[],col_dual=[],row_dual=[])

		#self.slave_prob[scen_number].write("dualSlaveCurrentLP.lp")
		
		algo = self.slave_prob[scen_number].parameters.lpmethod.values
		self.slave_prob[scen_number].parameters.lpmethod.set(algo.barrier)
		self.slave_prob[scen_number].parameters.barrier.crossover.set(self.slave_prob[scen_number].parameters.barrier.crossover.values.none)

		self.slave_prob[scen_number].solve()
		x = self.slave_prob[scen_number].solution.get_values()
		ob = self.slave_prob[scen_number].solution.get_objective_value()
		if(self.global_count == 0):
			return x,ob
		else:
			if(parallel_option == "mthread"):
				x.append(ob)
				x.insert(0,scen_number)
				return x
			if(parallel_option == "mprocess"):
				x.append(ob)
				x.insert(0,scen_number)
				send_end.send(x)


def main(argv):

	data_file_location = "/Users/supriyo/Box Sync/Projetcs/PortfolioOptimization/StocProgramCplex/data/json_samples/scenario_snp15_1.json"
	config_file = "../config/config_snp15_v1.txt"
	init_alloc = [0 for i in range(15)]
	init_alloc.append(100)
	result_file = "../Results/results_version1.txt"
	probInstance = stocProgPH(data_file_location, config_file, init_alloc) # pass horizon_length, num_assets, initial_values (x[0]), selling transc alpha_plus, buying transc alpha_minus
	probInstance.solveStocProgMaster()


if __name__ == "__main__":
	main(sys.argv[1:])
