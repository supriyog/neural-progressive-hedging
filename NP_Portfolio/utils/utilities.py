"""
@author: Supriyo GHOSH (IBM Research AI, Singapore)
Details: Find the solution of a stochastic program for Liquidity Portfolio Optimization
Dated: 27-11-2019
"""

import random
import sys
import math
import copy
import os
import numpy as np
from datetime import datetime
import json
from anytree import Node, RenderTree
from bidict import bidict
import pandas as pd
import datetime as dt

eps = 1e-8

@staticmethod
def sharpe(returns, freq=30, rfr=0):
	""" Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
	return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


@staticmethod
def max_drawdown(returns):
	""" Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
	peak = returns.max()
	trough = returns[returns.argmax():].min()
	return (trough - peak) / (peak + eps)

def write_results(string_value, result_file_name):
	fwrite = ""
	if os.path.exists(result_file_name):
		fwrite = open(result_file_name, "a")
	else:
		fwrite = open(result_file_name, "w")
	fwrite.write(string_value + "\n")
	fwrite.close()
	print (string_value)

def readJasonScenarioFile(filename):

	with open(filename) as json_file:
		data_json = json.load(json_file)

	scen_tree_nodes = bidict({})
	node_data = {}
	cur_node_num = 0
	cur_time_step = 1

	scen_tree_nodes[0] = Node("0")
	node_data[0] = {}
	for label_id in data_json.keys():
		if(label_id != "children"):
			node_data[0][label_id] = data_json[label_id]

	active_data = {}
	active_data[0] = data_json["children"]
	active_number_node = [nid for nid in range(len(active_data[0]))]
	while len(active_number_node) != 0 :
		active_number_node = []
		next_children_info = {}
		for parent_nodes in active_data.keys():
			for new_node in range(len(active_data[parent_nodes])):
				cur_node_num += 1
				scen_tree_nodes[cur_node_num] = Node(str(cur_node_num), parent = scen_tree_nodes[parent_nodes])
				node_data[cur_node_num] = {}
				for label_id in active_data[parent_nodes][new_node].keys():
					if(label_id != "children"):
						node_data[cur_node_num][label_id] = active_data[parent_nodes][new_node][label_id]
					if (label_id == "time_step"):
						if(active_data[parent_nodes][new_node][label_id] != cur_time_step):	# sanity check
							print ("something wrong "+str(active_data[parent_nodes][new_node][label_id])+" "+str(cur_time_step))
				if("children" in active_data[parent_nodes][new_node].keys()):
					next_children_info[cur_node_num] = active_data[parent_nodes][new_node]["children"]
					active_number_node.append(len(next_children_info[cur_node_num]))

		active_data = copy.deepcopy(next_children_info)
		cur_time_step += 1

	return scen_tree_nodes, node_data

def get_common_parents(scen1, scen2, scenario_nodes):
	for node1 in range(len(scenario_nodes[scen1])-1, -1, -1):
		for node2 in range(len(scenario_nodes[scen2])-1, -1, -1):
			if(scenario_nodes[scen1][node1] == scenario_nodes[scen2][node2]):
				return scenario_nodes[scen1][node1]
	#print (scenario_nodes[scen1][node1], scenario_nodes[scen2][node2])

def getSiblingScenarios(scenario_nodes, horizon_length, node_level, scen_tree):
	num_scens = len(scenario_nodes.keys())
	sibling_scens = [[0 for i in range(horizon_length)] for j in range(num_scens)]
	common_parents = [[0 for i in range(num_scens)] for j in range(num_scens)]

	for scen1 in range(num_scens):
		for scen2 in range(scen1,num_scens):
			if(scen1==scen2):
				common_parents[scen1][scen2] = horizon_length-1
			else:
				common_parents[scen1][scen2] = node_level[get_common_parents(scen1, scen2, scenario_nodes)]
				common_parents[scen2][scen1] = common_parents[scen1][scen2]

	for tstep in range(horizon_length):
		for scen in range(num_scens):
			if ((scen < num_scens-1) and (common_parents[scen][scen+1] >= tstep)):
				sibling_scens[scen][tstep] = scen+1
			else:
				for scen1 in range(num_scens):
					if(common_parents[scen][scen1] >= tstep):
						sibling_scens[scen][tstep] = scen1
						break
	return sibling_scens


def get_data(start_date, end_date, stocks):
	pathname_for_stock = 'sandp500/individual_stocks_5yr/individual_stocks_5yr/{}_data.csv'  ##must fill in symbol

	#ref_date = dt.date(2013, 2, 8)
	ref_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()

	abbr = []
	feats = []
	days = []
	for symb in stocks:
		x = pd.read_csv(pathname_for_stock.format(symb))
		mask = (x['date'] >= start_date) & (x['date'] <= end_date)
		x = x.loc[mask]
		z = {}
		z['day'] = np.array([(dt.datetime.strptime(s, '%Y-%m-%d').date() - ref_date).days for s in x['date']])
		z['open'] = x['open'].values
		z['high'] = x['high'].values
		z['low'] = x['low'].values
		z['close'] = x['close'].values
		z['volume'] = x['volume'].values
		f = np.array([z['day'], z['open'], z['high'], z['low'], z['close'], z['volume']]).T
		if np.isnan(f).any():
			print('skip file with NaN:', symb)
			continue
		feats.append(f)
		abbr.append(symb)
		if len(days) > 0:
			days = np.intersect1d(days, z['day'])
		else:
			days = z['day']

	prices = []
	for f in feats:
		idx = np.nonzero(np.isin(f[:, 0], days))[0]
		prices.append(f[idx, 1:])

	return np.array(prices), abbr


def generate_scenario_tree(look_ahead_period,n_sample_scenario,interest_samples,cash_return, json_write_file):
	num_asset = interest_samples[0][0].shape[1]
	print (num_asset)
	root_node = {"time_step": 0, "interest_rate": ' '.join(map(str,[0 for i in range(num_asset)])), "zc_price": 0, "liability": 0}
	root_node["children"] = []
	parent_nodes = [root_node]
	for ts in range(1,look_ahead_period):
		new_parent_nodes = []
		parent_count = 0
		for p in parent_nodes:
			for s in range(n_sample_scenario[ts-1]):
				cur_child = {"time_step": ts, "zc_price": cash_return, "liability": 0}
				cur_child["interest_rate"] = ' '.join(map(str, interest_samples[ts-1][parent_count][s]))
				if(ts < look_ahead_period-1):
					cur_child["children"] = []
				p["children"].append(cur_child)
				new_parent_nodes.append(cur_child)
			parent_count += 1
		parent_nodes = new_parent_nodes

	file_write = json_write_file#"../Dataset/scenario_snp15_1.json"
	with open(file_write, "w") as write_file:
		json.dump(root_node, write_file)

	return file_write


def reorderScenarios():
	pass		# already sorted according to our need
