import networkx as nx
import random
from random import sample
import matplotlib.pyplot as plt
import copy
import pickle
import os
from math import ceil
from sklearn.linear_model import LogisticRegression

from utils import *
from networkx.algorithms.link_prediction import adamic_adar_index, preferential_attachment

class Graph:
	def __init__(self, G):
		"""
		Initialize a NetworkX graph from a file with edge list.
		Raises Exception if provided file is not an edge list
		"""
		

		lt = []
		for i in G.nodes():
			lt.append((i,i))
		G.remove_edges_from(lt)

		self.G = nx.convert_node_labels_to_integers(G)
		self.vertex_set = set(self.G.nodes())
		self.vertex_list = list(self.vertex_set)
		self.num_nodes = max(self.vertex_set)+1
		# remove self loops

	def split_train_test(self, test_fraction):
		"""
		Prepares the graph for training by creating a train, test graph with non-overlapping edges 
		Input test_fraction: Fraction of neighbours per node that make the test split.
		Returns: None
		Adds to the self test_edges_list, train_edges_list both of which are list of triples (in, out, edge-type)
		A new graph with edges from test omitted is attached to self called G_train. 
		"""
		assert test_fraction<=1 and test_fraction>=0

		self.test_fraction = test_fraction
		
		nbr_dict, nonnbr_dict = find_nbr_nonnbr(self.G)
		self.nbr_dict, self.nonnbr_dict = nbr_dict, nonnbr_dict
		
		per_node_train_set, per_node_test_set = {}, {}		   
		test_edges_list, train_edges_list = [], []		
		for node in range(len(self.vertex_list)):
			per_node_test_set[node], per_node_train_set[node] = {}, {}
			
			x_nbr = int(test_fraction*len(nbr_dict[node]))
			x_nonnbr = int(test_fraction*len(nonnbr_dict[node]))
			
			per_node_test_set[node]['nbr'] = sample(nbr_dict[node], x_nbr)
			per_node_train_set[node]['nbr'] =  list(set(nbr_dict[node])\
													   - set(per_node_test_set[node]['nbr']))
	
			per_node_test_set[node]['nonnbr'] = sample(nonnbr_dict[node], x_nonnbr)
			per_node_train_set[node]['nonnbr'] =  list(set(nonnbr_dict[node])\
												  - set(per_node_test_set[node]['nonnbr']))
	

			test_edges_per_node = [(node, x) for x in per_node_test_set[node]['nbr']]
			test_non_edges_per_node  = [(node, x) for x in per_node_test_set[node]['nonnbr']]
			train_edges_per_node = [(node, x) for x in per_node_train_set[node]['nbr']]
			train_non_edges_per_node  = [(node, x) for x in per_node_train_set[node]['nonnbr']]
			
			test_edges_list.extend([(a, b, 1) for a, b in test_edges_per_node if a < b])
			test_edges_list.extend([(a, b, 0) for a, b in test_non_edges_per_node if a < b])

			train_edges_list.extend([(a, b, 1) for a, b in train_edges_per_node if a < b])
			train_edges_list.extend([(a, b, 0) for a, b in train_non_edges_per_node if a < b])
			
		self.test_edges_list = test_edges_list
		self.train_edges_list = train_edges_list

		G_train =  copy.deepcopy(self.G)
		G_train.remove_edges_from([(a, b) for (a, b, label) in test_edges_list if label==1])
		self.G_train = G_train

class myLogisticRegression(object):
	"""docstring for myLogisticRegression"""
	def __init__(self, graph, test_fraction, ebunch, num_train_sets = 1, load_pickle = False, dump_pickle = False):
		super(myLogisticRegression, self).__init__()
		self.corelr = LogisticRegression(random_state = 0, max_iter = 500, solver = 'liblinear', penalty = 'l1')
		self.ebunch = ebunch
		self.backbrop_test_edges_nonedges = test_edges_nonedges = [(u,v) for (u,v,_) in graph.test_edges_list]
		self.dump_pickle = dump_pickle
		self.load_pickle = load_pickle
		self.graph = graph
		self.test_fraction = test_fraction
		self.num_train_sets = num_train_sets
		self.prev_seed = 0
		self.aa = {}
		self.pa = {}
		self.cn = {}
		self.km = {}

	def regenerate_graph(self):
		print('Regenerating Graph')
		self.prev_seed += 1
		random.seed(self.prev_seed)
		self.graph.split_train_test(self.test_fraction)

	def regenerate_features(self):
		if self.load_pickle:
			with open('./pickles/lr_train.pickle', 'wb') as handle:
				dic = pickle.load(handle)
			self.aa = dic['aa']
			self.pa = dic['pa']
			self.cn = dic['cn']
			self.km = dic['km']
		else:
			print('\nGenerating Train Features')
			print('Generating Adamic Adar')
			aalist = list(adamic_adar_index(self.graph.G_train, ebunch = self.backbrop_test_edges_nonedges))
			print('Generating Common Neighbor')
			cnlist = list(common_neighbor(self.graph.G_train, ebunch = self.backbrop_test_edges_nonedges))
			print('Generating Preferential Attachment')
			palist = list(preferential_attachment(self.graph.G_train, ebunch = self.backbrop_test_edges_nonedges))
			print('Generating Katz Measure')
			kmlist = katz_measure(self.graph.G_train, forgetting_factor_scale = .5, ebunch = self.backbrop_test_edges_nonedges)

			self.aa = {(a,b):c for (a,b,c) in aalist}
			self.pa = {(a,b):c for (a,b,c) in palist}
			self.cn = {(a,b):c for (a,b,c) in cnlist}
			self.km = {(a,b):c for (a,b,c) in kmlist}

			dic = {}
			dic['aa'] = self.aa
			dic['pa'] = self.pa
			dic['cn'] = self.cn
			dic['km'] = self.km
			if self.dump_pickle:
				with open('./pickles/lr_train.pickle', 'wb') as handle:
					pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print('Populating Features')
		num_x = len(self.backbrop_test_edges_nonedges)
		self.X = np.ones((num_x,5))
		self.Y = np.ones((num_x,))
		self.num_edge = 0
		self.num_nonedge = 0
		for i, (a,b,label) in enumerate(self.graph.test_edges_list):
			self.X[i,1] = self.aa[(a,b)]
			self.X[i,2] = self.pa[(a,b)]
			self.X[i,3] = self.cn[(a,b)]
			self.X[i,4] = self.km[(a,b)]
			self.Y[i] = label
			if label == 1:
				self.num_edge += 1
			else:
				self.num_nonedge += 1

		si = np.random.shuffle(np.arange(self.X.shape[0]))
		self.X = self.X[si,:].squeeze(0)
		self.Y = self.Y[si].squeeze(0)

		Xmean = self.X.mean(0).reshape(1,-1)
		Xstd = self.X.std(0).reshape(1,-1)
		Xstd[:,0] = 1
		self.X = (self.X - Xmean)/Xstd

	def generate_test_features(self):
		if self.load_pickle:
			with open('./pickles/lr_test.pickle', 'wb') as handle:
				dic = pickle.load(handle)
			self.aa = dic['aa']
			self.pa = dic['pa']
			self.cn = dic['cn']
			self.km = dic['km']
		else:
			print('\nGenerating test Features')
			print('Generating Adamic Adar')
			aalist = list(adamic_adar_index(self.graph.G_train, ebunch = self.ebunch))
			print('Generating Common Neighbor')
			cnlist = list(common_neighbor(self.graph.G_train, ebunch = self.ebunch))
			print('Generating Preferential Attachment')
			palist = list(preferential_attachment(self.graph.G_train, ebunch = self.ebunch))
			print('Generating Katz Measure\n')
			kmlist = katz_measure(self.graph.G_train, forgetting_factor_scale = .5, ebunch = self.ebunch)

			self.aa = {(a,b):c for (a,b,c) in aalist}
			self.pa = {(a,b):c for (a,b,c) in palist}
			self.cn = {(a,b):c for (a,b,c) in cnlist}
			self.km = {(a,b):c for (a,b,c) in kmlist}

			dic = {}
			dic['aa'] = self.aa
			dic['pa'] = self.pa
			dic['cn'] = self.cn
			dic['km'] = self.km
			if self.dump_pickle:
				with open('./pickles/lr_test.pickle', 'wb') as handle:
					pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


		self.X_test = np.zeros((len(self.ebunch), 5))
		for i, (a,b) in enumerate(self.ebunch):
			self.X_test[i,1] = self.aa[(a,b)]
			self.X_test[i,2] = self.pa[(a,b)]
			self.X_test[i,3] = self.cn[(a,b)]
			self.X_test[i,4] = self.km[(a,b)]

		Xmean = self.X_test.mean(0).reshape(1,-1)
		Xstd = self.X_test.std(0).reshape(1,-1)
		Xstd[:,0] = 1
		self.X_test = (self.X_test - Xmean)/Xstd

	def get_scores(self):
		self.generate_test_features()
		scores = self.corelr.decision_function(self.X_test)

		ans = []
		for i,(a,b) in enumerate(self.ebunch):
			ans.append((a,b,scores[i]))
		ans.sort(reverse = True, key=lambda x:x[2])

		return ans

	def train(self):
		for i in range(self.num_train_sets):
			print('Training on set {}'.format(i))
			if not i == 0:
				self.regenerate_graph()
			self.regenerate_features()
			self.corelr.fit(self.X, self.Y)

		