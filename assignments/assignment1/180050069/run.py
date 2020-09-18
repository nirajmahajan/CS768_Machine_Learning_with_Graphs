import os
import copy
import random
import argparse
import networkx as nx
import pickle

from random import sample
from classes import Graph
from utils import MAP, MRR, katz_measure, common_neighbor
from networkx.algorithms.link_prediction import adamic_adar_index, preferential_attachment

# constant declarations
filename = 'datasets/facebook.txt'

# Arg parser declarations
parser = argparse.ArgumentParser()
parser.add_argument('--only_adamic_adar', action = 'store_true')
parser.add_argument('--only_preferential_attachment', action = 'store_true')
parser.add_argument('--only_katz_measure', action = 'store_true')
parser.add_argument('--only_common_neigbours', action = 'store_true')
parser.add_argument('--only_logistic_regression', action = 'store_true')
parser.add_argument('--load_pickle', action = 'store_true')
parser.add_argument('--test_fraction', type = float, default = 0.2)
args = parser.parse_args()

just_AA = False
just_PA = False
just_KM = False
just_CN = False
just_LR = False

if args.only_adamic_adar:
	just_AA = True
if args.only_preferential_attachment:
	just_PA = True
if args.only_katz_measure:
	just_KM = True
if args.only_common_neigbours:
	just_CN = True
if args.only_logistic_regression:
	just_LR = True
if not (just_CN or just_KM or just_PA or just_AA or just_LR):
	just_AA = True
	just_PA = True
	just_KM = True
	just_CN = True
	just_LR = True

random.seed(0)
test_fraction_mapping = {}
test_fraction_mapping[0.1] = 0.062
test_fraction_mapping[0.2] = 0.117
test_fraction_mapping[0.3] = 0.174
test_fraction_mapping[0.4] = 0.237

print('Desired Test Fraction		  = {}'.format(args.test_fraction))
print('Using Test Fraction (for code) = {}\n'.format(test_fraction_mapping[args.test_fraction]))


if args.load_pickle:
	with open('./pickles/graph.pickle', 'rb') as handle:
		graph = pickle.load(handle)['graph']
else:
	graph = Graph(filename)
	graph.split_train_test(test_fraction_mapping[args.test_fraction])

	with open('./pickles/graph.pickle', 'wb') as handle:
		pickle.dump({'graph':graph}, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
print ("Observed fraction of train non edges	  : %0.4f\n" % (len(graph.G_train_invert.edges())/(len(graph.G_train_invert.edges()) + len(graph.G_test_invert.edges()))))

for (a,b) in graph.G_test.edges():
	if (a == b):
		print("Self Loop detected in test edges. Terminating.")
		os._exit(0)

for (a,b) in graph.G_test_invert.edges():
	if (a == b):
		print("Self Loop detected in test non edges. Terminating.")
		os._exit(0)

for (a,b) in graph.G_train.edges():
	if (a == b):
		print("Self Loop detected in train edges. Terminating.")
		os._exit(0)

for (a,b) in graph.G_train_invert.edges():
	if (a == b):
		print("Self Loop detected in train non edges. Terminating.")
		os._exit(0)

test_edges_nonedges = list(graph.G_test_invert.edges()) + list(graph.G_test.edges())
test_labels = {}
for (a, b) in graph.G_test.edges():
	if a > b:
		temp = a
		a = b
		b = temp
	test_labels[(a,b)] = 1
for (a, b) in graph.G_test_invert.edges():
	if a > b:
		temp = a
		a = b
		b = temp
	test_labels[(a,b)] = 0


if just_AA:
	print('Adamic Adar', flush = True)
	if args.load_pickle:
		with open('./pickles/aa.pickle', 'rb') as handle:
			aalist = pickle.load(handle)['aalist']
	else:
		aalist = list(adamic_adar_index(graph.G_train, ebunch = test_edges_nonedges))
		with open('./pickles/aa.pickle', 'wb') as handle:
			pickle.dump({'aalist':aalist}, handle, protocol=pickle.HIGHEST_PROTOCOL)

	map = MAP(aalist, test_labels, graph.num_nodes)
	mrr = MRR(aalist, test_labels, graph.num_nodes)
	print('MAP={}'.format(map))
	print('MRR={}'.format(mrr))
	print('')

if just_CN:
	print('Common Neighbor', flush = True)
	if args.load_pickle:
		with open('./pickles/cn.pickle', 'rb') as handle:
			cnlist = pickle.load(handle)['cnlist']
	else:
		cnlist = list(common_neighbor(graph.G_train, ebunch = test_edges_nonedges))
		with open('./pickles/cn.pickle', 'wb') as handle:
			pickle.dump({'cnlist':cnlist}, handle, protocol=pickle.HIGHEST_PROTOCOL)

	map = MAP(cnlist, test_labels, graph.num_nodes)
	mrr = MRR(cnlist, test_labels, graph.num_nodes)
	print('MAP={}'.format(map))
	print('MRR={}'.format(mrr))
	print('')

if just_PA:
	print('Preferential Attachment', flush = True)
	if args.load_pickle:
		with open('./pickles/pa.pickle', 'rb') as handle:
			palist = pickle.load(handle)['palist']
	else:
		palist = list(preferential_attachment(graph.G_train, ebunch = test_edges_nonedges))
		with open('./pickles/pa.pickle', 'wb') as handle:
			pickle.dump({'palist':palist}, handle, protocol=pickle.HIGHEST_PROTOCOL)

	map = MAP(palist, test_labels, graph.num_nodes)
	mrr = MRR(palist, test_labels, graph.num_nodes)
	print('MAP={}'.format(map))
	print('MRR={}'.format(mrr))
	print('')

if just_KM:
	print('Katz', flush = True)
	if args.load_pickle:
		with open('./pickles/km.pickle', 'rb') as handle:
			kmlist = pickle.load(handle)['kmlist']
	else:
		kmlist = katz_measure(graph.G_train, forgetting_factor_scale = .5, ebunch = test_edges_nonedges)
		with open('./pickles/km.pickle', 'wb') as handle:
			pickle.dump({'kmlist':kmlist}, handle, protocol=pickle.HIGHEST_PROTOCOL)
	map = MAP(kmlist, test_labels, graph.num_nodes)
	mrr = MRR(kmlist, test_labels, graph.num_nodes)
	print('MAP={}'.format(map))
	print('MRR={}'.format(mrr))

if just_LR:
	 print('Training Logistic Regression')
	 lr_instance = LogisticRegression()
	 indexlst = list(preferential_attachment(graph.G_train))
	 map = MAP(indexlst, test_labels, graph.num_nodes)
	 mrr = MRR(indexlst, test_labels, graph.num_nodes)
	 print('MAP={}'.format(map))
	 print('MRR={}'.format(mrr))