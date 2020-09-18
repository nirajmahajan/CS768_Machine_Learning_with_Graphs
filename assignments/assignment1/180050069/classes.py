import networkx as nx
import random
from random import sample
import copy
import os

from utils import *

class Graph:
	def __init__(self, filename):
		"""
		Initialize a NetworkX graph from a file with edge list.
		Raises Exception if provided file is not an edge list
		"""
		G = nx.read_edgelist(filename)

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
	
			include_in_test = True
			if not per_node_test_set[node]['nbr']:
				include_in_test = False
			if not per_node_test_set[node]['nonnbr']:
				include_in_test = False

			test_edges_per_node = [(node, x) for x in per_node_test_set[node]['nbr']]
			test_non_edges_per_node  = [(node, x) for x in per_node_test_set[node]['nonnbr']]
			train_edges_per_node = [(node, x) for x in per_node_train_set[node]['nbr']]
			train_non_edges_per_node  = [(node, x) for x in per_node_train_set[node]['nonnbr']]
			
			if include_in_test:
				test_edges_list.extend([(a, b, 1) for a, b in test_edges_per_node if not a == b])
				test_edges_list.extend([(a, b, 0) for a, b in test_non_edges_per_node if not a == b])
			else:
				train_edges_list.extend([(a, b, 1) for a, b in test_edges_per_node if not a == b])
				train_edges_list.extend([(a, b, 0) for a, b in test_non_edges_per_node if not a == b])

			train_edges_list.extend([(a, b, 1) for a, b in train_edges_per_node if not a == b])
			train_edges_list.extend([(a, b, 0) for a, b in train_non_edges_per_node if not a == b])
			
		# self.test_edges_list = test_edges_list
		# self.train_edges_list = train_edges_list

		# for (a,b,label) in self.test_edges_list:
		#	 if (a == b):
		#		 self.test_edges_list.remove((a,b,label))

		# for (a,b,label) in self.train_edges_list:
		#	 if (a == b):
		#		 print(a,b)
		#		 self.train_edges_list.remove((a,b,label))
		
		train_non_edges_exc = [(a,b) for (a,b,label) in train_edges_list if label == 0]
		test_non_edges_exc = [(a,b) for (a,b,label) in test_edges_list if label == 0]

		G_train_invert = nx.Graph()
		G_train_invert.add_edges_from(train_non_edges_exc)
		G_test_invert = nx.Graph()
		G_test_invert.add_edges_from(test_non_edges_exc)

		self.G_train_invert = G_train_invert
		self.G_test_invert = G_test_invert

		G_train =  copy.deepcopy(self.G)
		G_test =  copy.deepcopy(self.G)
		G_train.remove_edges_from([(a, b) for (a, b, label) in test_edges_list if label==1])
		G_test.remove_edges_from(G_train.edges())
		self.G_train = G_train
		self.G_test = G_test

class LogisticRegression(object):
	"""docstring for LogisticRegression"""
	def __init__(self, test_edge_list, aa, pa, cn, km, testlabels):
		super(LogisticRegression, self).__init__()
		self.test_edge_list = test_edge_list
		self.n_samples = (len(G_train.edges))
		self.batch_size = 500
		self.n_iterations = ceil(self.n_samples/self.batch_size)
		self.X = np.ones((self.n_samples, 5))
		self.Y = np.ones((self.n_samples, 1))
		self.W = np.random.normal(size=(5,))/(self.X.shape[0] ** 0.5)
		self.aa = {(a,b):c for (a,b,c) in aa}
		self.pa = {(a,b):c for (a,b,c) in pa}
		self.cn = {(a,b):c for (a,b,c) in cn}
		self.km = {(a,b):c for (a,b,c) in km}

	def populate_features(self):
		for i, (a,b, label) in enumerate(self.test_edge_list):
			self.X[i,1] = self.aa[(a,b)]
			self.X[i,2] = self.pa[(a,b)]
			self.X[i,3] = self.cn[(a,b)]
			self.X[i,4] = self.km[(a,b)]
			self.Y[i,0] = label

	def train(self, epochs = 100, lr = 0.01):
		for e in range(epochs):
			for batch_num in range(self.n_iterations):
				if batch_num == self.n_iterations-1:
					X_batch = self.X[batch_num*self.batch_size:,:]
					Y_batch = self.Y[batch_num*self.batch_size:]
				else:
					X_batch = self.X[batch_num*self.batch_size:(batch_num+1)*self.batch_size,:]
					Y_batch = self.Y[batch_num*self.batch_size:(batch_num+1)*self.batch_size]

				Y_hat = np.matmul(X_batch, self.W)

				gradient = np.dot(X.T,  predictions - labels)/5

				self.W = self.W - lr*gradient

	def get_scores(self, ebunch):
		ans = []
		if not ebunch is None:
			for (a,b) in ebunch:
				X_temp = np.zeros((1,5))
				X_temp[0,1] = aa[(a,b)]
				X_temp[0,2] = pa[(a,b)]
				X_temp[0,3] = cn[(a,b)]
				X_temp[0,4] = km[(a,b)]
				ans.append((a,b,yhat))
		else:
			for a in range(max_size):
				for b in range(max_size):
					if not (a,b) in gtrain.edges():
						ans.append((a,b,S[a,b]))
		ans.sort(reverse = True, key=lambda x:x[2])
		return ans

		