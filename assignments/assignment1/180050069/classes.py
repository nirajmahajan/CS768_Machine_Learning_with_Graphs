import networkx as nx
import random
from random import sample
import matplotlib.pyplot as plt
import copy
import pickle
import os
from math import ceil
from scipy.special import expit as sigmoid


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
				test_edges_list.extend([(a, b, 1) for a, b in test_edges_per_node if a < b])
				test_edges_list.extend([(a, b, 0) for a, b in test_non_edges_per_node if a < b])
			else:
				train_edges_list.extend([(a, b, 1) for a, b in test_edges_per_node if a < b])
				train_edges_list.extend([(a, b, 0) for a, b in test_non_edges_per_node if a < b])

			train_edges_list.extend([(a, b, 1) for a, b in train_edges_per_node if a < b])
			train_edges_list.extend([(a, b, 0) for a, b in train_non_edges_per_node if a < b])
			
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
	def __init__(self, graph_train_validation, test_fraction, ebunch, num_train_sets = 1, load = False):
		super(LogisticRegression, self).__init__()
		self.ebunch = ebunch
		self.load = load
		self.graph_train_validation = graph_train_validation
		self.test_fraction = test_fraction
		self.num_train_sets = num_train_sets
		self.prev_seed = 0
		if load:
			load_weights()
		else:
			self.W = np.random.normal(size = (5,1)) * 0.001
		# self.W = np.array([[0],[1],[0],[0],[0]]).astype(np.float)
		self.aa = {}
		self.pa = {}
		self.cn = {}
		self.km = {}

	def regenerate_graph(self):
		print('Regenerating Graph')
		self.prev_seed += 1
		random.seed(self.prev_seed)
		self.graph_train_validation.split_train_test(self.test_fraction)

	def regenerate_features(self):
		print('Generating Features')
		test_edges_nonedges = list(self.graph_train_validation.G_test_invert.edges()) + list(self.graph_train_validation.G_test.edges())
		print('Generating Adamic Adar')
		aalist = list(adamic_adar_index(self.graph_train_validation.G_train, ebunch = test_edges_nonedges))
		print('Generating Common Neighbor')
		cnlist = list(common_neighbor(self.graph_train_validation.G_train, ebunch = test_edges_nonedges))
		print('Generating Preferential Attachment')
		palist = list(preferential_attachment(self.graph_train_validation.G_train, ebunch = test_edges_nonedges))
		print('Generating Katz Measure')
		kmlist = katz_measure(self.graph_train_validation.G_train, forgetting_factor_scale = .5, ebunch = test_edges_nonedges)

		self.aa = {(a,b):c for (a,b,c) in aalist}
		self.pa = {(a,b):c for (a,b,c) in palist}
		self.cn = {(a,b):c for (a,b,c) in cnlist}
		self.km = {(a,b):c for (a,b,c) in kmlist}


	def populate_features(self):
		print('Populating Features')
		num_x = len(self.graph_train_validation.G_test.edges()) + len(self.graph_train_validation.G_test_invert.edges())
		self.X = np.ones((num_x,5))
		self.Y = np.ones((num_x,1))
		self.num_edge = 0
		self.num_nonedge = 0
		for i, (a,b) in enumerate(self.graph_train_validation.G_test.edges()):
			self.X[i,1] = self.aa[(a,b)]
			self.X[i,2] = self.pa[(a,b)]
			self.X[i,3] = self.cn[(a,b)]
			self.X[i,4] = self.km[(a,b)]
			self.Y[i,0] = 1
			self.num_edge += 1
		for i, (a,b) in enumerate(self.graph_train_validation.G_test_invert.edges()):
			self.X[i,1] = self.aa[(a,b)]
			self.X[i,2] = self.pa[(a,b)]
			self.X[i,3] = self.cn[(a,b)]
			self.X[i,4] = self.km[(a,b)]
			self.Y[i,0] = 0
			self.num_nonedge += 1

		si = np.random.shuffle(np.arange(self.X.shape[0]))
		self.X = self.X[si,:].squeeze(0)
		self.Y = self.Y[si,:].squeeze(0)

		self.Xmean = self.X.mean(0).reshape(1,-1)
		self.Xstd = self.X.std(0).reshape(1,-1)
		self.Xstd[:,0] = 1
		self.X = (self.X - self.Xmean)/self.Xstd

	def train_set(self, epochs = 10000, lr = 1.):
		self.batch_size = 500
		self.n_iterations = ceil(self.X.shape[0]/self.batch_size)
		loss = []
		for e in range(epochs):
			with open('log.txt', 'a') as f:
				f.write("\nEpoch: {}\n".format(e))
			if e % 100 == 0:
				with open('./pickles/weights.pickle', 'wb') as handle:
					pickle.dump({'W':self.W}, handle, protocol=pickle.HIGHEST_PROTOCOL)
			totloss = 0
			for batch_num in range(self.n_iterations):
				if batch_num == self.n_iterations-1:
					X_batch = self.X[batch_num*self.batch_size:,:]
					Y_batch = self.Y[batch_num*self.batch_size:]
				else:
					X_batch = self.X[batch_num*self.batch_size:(batch_num+1)*self.batch_size,:]
					Y_batch = self.Y[batch_num*self.batch_size:(batch_num+1)*self.batch_size,:]

				edge_indices = Y_batch.reshape(-1) == 1
				nonedge_indices = Y_batch.reshape(-1) == 0

				# 500x1
				Y_hat = sigmoid(X_batch.dot(self.W))
				s0 = (Y_hat == 0).sum()
				s1 = (Y_hat == 1).sum()
				if s1 > 0:
					print(X_batch.dot(self.W).max())
				assert(s0 == 0)
				assert(s1 == 0)

				class1cost = -Y_batch*np.log(Y_hat)/self.num_edge
				class2cost = (1-Y_batch)*np.log(1-Y_hat)/self.num_nonedge
				cost = (class1cost - class2cost).sum()/self.batch_size
				totloss += cost

				gradient1  = np.dot(X_batch[edge_indices,:].T,  Y_hat[edge_indices] - Y_batch[edge_indices])/500
				gradient2  = np.dot(X_batch[nonedge_indices,:].T,  Y_hat[nonedge_indices] - Y_batch[nonedge_indices])/500
				gradient = (gradient1 / self.num_edge) + (gradient2 / self.num_nonedge)
				self.W = self.W - lr*gradient
			loss.append(totloss)
			with open('log.txt', 'a') as f:
				f.write("Loss: {}\n".format(totloss))
			# print(totloss)
		plt.figure()
		plt.plot(loss)
		plt.title("Loss for current training set")
		plt.show()

	def load_weights(self):
		with open('./pickles/weights.pickle', 'rb') as handle:
			self.W = pickle.load(handle)['W']

	def generate_test_features(self):
		print('Generating test Features \n')
		test_edges_nonedges = self.ebunch
		aalist = list(adamic_adar_index(self.graph_train_validation.G_train, ebunch = test_edges_nonedges))
		cnlist = list(common_neighbor(self.graph_train_validation.G_train, ebunch = test_edges_nonedges))
		palist = list(preferential_attachment(self.graph_train_validation.G_train, ebunch = test_edges_nonedges))
		kmlist = katz_measure(self.graph_train_validation.G_train, forgetting_factor_scale = .5, ebunch = test_edges_nonedges)

		self.aa = {(a,b):c for (a,b,c) in aalist}
		self.pa = {(a,b):c for (a,b,c) in palist}
		self.cn = {(a,b):c for (a,b,c) in cnlist}
		self.km = {(a,b):c for (a,b,c) in kmlist}

	def get_scores(self):
		self.generate_test_features()
		ans = []
		for (a,b) in self.ebunch:
			X_temp = np.ones((1,5))
			X_temp[0,1] = self.aa[(a,b)]
			X_temp[0,2] = self.pa[(a,b)]
			X_temp[0,3] = self.cn[(a,b)]
			X_temp[0,4] = self.km[(a,b)]
			X_temp = (X_temp - self.Xmean)/self.Xstd
			yhat = np.matmul(X_temp, self.W)[0][0]
			ans.append((a,b,yhat))
		ans.sort(reverse = True, key=lambda x:x[2])
		return ans

	def train(self):
		for i in range(self.num_train_sets):
			print('Training on set {}'.format(i))
			if not i == 0:
				self.regenerate_graph()
			self.regenerate_features()
			self.populate_features()
			self.train_set()

		