import os
import copy
import random
import argparse
import networkx as nx
import pickle

from random import sample
from classes import Graph, myLogisticRegression
from utils import MAP, MRR, katz_measure, common_neighbor
from networkx.algorithms.link_prediction import adamic_adar_index, preferential_attachment

# constant declarations
filename = 'datasets/facebook.txt'
pickle_path = './pickles/'

# Arg parser declarations
parser = argparse.ArgumentParser()
parser.add_argument('--only_adamic_adar', action = 'store_true')
parser.add_argument('--only_preferential_attachment', action = 'store_true')
parser.add_argument('--only_katz_measure', action = 'store_true')
parser.add_argument('--only_common_neighbors', action = 'store_true')
parser.add_argument('--only_logistic_regression', action = 'store_true')
parser.add_argument('--load_lr', action = 'store_true')
parser.add_argument('--load_pickle', action = 'store_true')
parser.add_argument('--mini', action = 'store_true')
parser.add_argument('--test_fraction', type = float, default = 0.2)
args = parser.parse_args()

if args.mini:
	filename = 'datasets/facebookmini.txt'
	pickle_path = './minipickles/'
	
just_AA = False
just_PA = False
just_KM = False
just_CN = False
just_LR = False

topk = None

if args.only_adamic_adar:
	just_AA = True
if args.only_preferential_attachment:
	just_PA = True
if args.only_katz_measure:
	just_KM = True
if args.only_common_neighbors:
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
test_fraction_mapping[0.1] = 0.11
test_fraction_mapping[0.2] = 0.2
test_fraction_mapping[0.3] = 0.309
test_fraction_mapping[0.4] = 0.41

test_fraction_mapping2 = 0.21

print('Desired Test Fraction		  = {}'.format(args.test_fraction))
print('Using Test Fraction (for code) = {}'.format(test_fraction_mapping[args.test_fraction]))
print('Using Test Fraction (for validation) = {}\n'.format(test_fraction_mapping2))


if args.load_pickle:
	with open(pickle_path + 'graph.pickle', 'rb') as handle:
		dic = pickle.load(handle)
		graph = dic['graph']
		gtrain_validate = dic['gtrain_validate']
else:
	G = nx.read_edgelist(filename)
	graph = Graph(G)
	print('Splitting Train-Test data')
	graph.split_train_test(test_fraction_mapping[args.test_fraction])
	newg = copy.deepcopy(graph.G_train)
	gtrain_validate = Graph(newg)
	print('Splitting Train-Validation data\n')
	gtrain_validate.split_train_test(test_fraction_mapping2)

	with open(pickle_path + 'graph.pickle', 'wb') as handle:
		pickle.dump({'graph':graph, 'gtrain_validate': gtrain_validate}, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ("Observed fraction of train/validation edges	  : %0.4f" % (len(gtrain_validate.G_train.edges())/(len(gtrain_validate.G.edges()))))
print ("Observed fraction of train edges	  : %0.4f\n" % (len(graph.G_train.edges())/(len(graph.G.edges()))))

test_edges_nonedges = [(u,v) for (u,v,_) in graph.test_edges_list]

test_labels = {}
for (a, b, label) in graph.test_edges_list:
	if a > b:
		temp = a
		a = b
		b = temp
	test_labels[(a,b)] = label

if just_AA:
	print('Adamic Adar', flush = True)
	if args.load_pickle:
		with open(pickle_path + 'aa.pickle', 'rb') as handle:
			aalist = pickle.load(handle)['aalist']
	else:
		aalist = list(adamic_adar_index(graph.G_train, ebunch = test_edges_nonedges))
		with open(pickle_path + 'aa.pickle', 'wb') as handle:
			pickle.dump({'aalist':aalist}, handle, protocol=pickle.HIGHEST_PROTOCOL)

	map = MAP(aalist, test_labels, graph.num_nodes, topk = topk)
	mrr = MRR(aalist, test_labels, graph.num_nodes, topk = topk)
	print('MAP={}'.format(map))
	print('MRR={}'.format(mrr))
	print('')

if just_CN:
	print('Common Neighbor', flush = True)
	if args.load_pickle:
		with open(pickle_path + 'cn.pickle', 'rb') as handle:
			cnlist = pickle.load(handle)['cnlist']
	else:
		cnlist = list(common_neighbor(graph.G_train, ebunch = test_edges_nonedges))
		with open(pickle_path + 'cn.pickle', 'wb') as handle:
			pickle.dump({'cnlist':cnlist}, handle, protocol=pickle.HIGHEST_PROTOCOL)

	map = MAP(cnlist, test_labels, graph.num_nodes, topk = topk)
	mrr = MRR(cnlist, test_labels, graph.num_nodes, topk = topk)
	print('MAP={}'.format(map))
	print('MRR={}'.format(mrr))
	print('')

if just_PA:
	print('Preferential Attachment', flush = True)
	if args.load_pickle:
		with open(pickle_path + 'pa.pickle', 'rb') as handle:
			palist = pickle.load(handle)['palist']
	else:
		palist = list(preferential_attachment(graph.G_train, ebunch = test_edges_nonedges))
		with open(pickle_path + 'pa.pickle', 'wb') as handle:
			pickle.dump({'palist':palist}, handle, protocol=pickle.HIGHEST_PROTOCOL)

	map = MAP(palist, test_labels, graph.num_nodes, topk = topk)
	mrr = MRR(palist, test_labels, graph.num_nodes, topk = topk)
	print('MAP={}'.format(map))
	print('MRR={}'.format(mrr))
	print('')

if just_KM:
	print('Katz', flush = True)
	if args.load_pickle:
		with open(pickle_path + 'km.pickle', 'rb') as handle:
			kmlist = pickle.load(handle)['kmlist']
	else:
		kmlist = katz_measure(graph.G_train, forgetting_factor_scale = .5, ebunch = test_edges_nonedges)
		with open(pickle_path + 'km.pickle', 'wb') as handle:
			pickle.dump({'kmlist':kmlist}, handle, protocol=pickle.HIGHEST_PROTOCOL)
	map = MAP(kmlist, test_labels, graph.num_nodes, topk = topk)
	mrr = MRR(kmlist, test_labels, graph.num_nodes, topk = topk)
	print('MAP={}'.format(map))
	print('MRR={}'.format(mrr))
	print('')

if just_LR:
	print('Training Logistic Regression')
	if args.load_pickle:
		with open(pickle_path + 'lr.pickle', 'rb') as handle:
			lrlist = pickle.load(handle)['lrlist']
	else:
		lr_instance = myLogisticRegression(graph, test_fraction_mapping2, test_edges_nonedges, num_train_sets = 1, load = args.load_lr)
		lr_instance.train()
		lrlist = lr_instance.get_scores()
		with open(pickle_path + 'lr.pickle', 'wb') as handle:
			pickle.dump({'lrlist':lrlist}, handle, protocol=pickle.HIGHEST_PROTOCOL)

	map = MAP(lrlist, test_labels, graph.num_nodes, topk = topk)
	mrr = MRR(lrlist, test_labels, graph.num_nodes, topk = topk)
	print('MAP={}'.format(map))
	print('MRR={}'.format(mrr))