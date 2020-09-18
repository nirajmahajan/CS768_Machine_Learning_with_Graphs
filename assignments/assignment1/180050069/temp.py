import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from random import sample
from classes import Graph
from utils import MAP, MRR, katz_measure, common_neighbor
from networkx.algorithms.link_prediction import adamic_adar_index, preferential_attachment

G = nx.Graph()
G.add_edge(0,2)
G.add_edge(1,2)
G.add_edge(1,3)
G.add_edge(1,4)


def katz_measure(gtrain, forgetting_factor_scale = 0.9, ebunch = None):
	max_size = max(gtrain.nodes())+1
	adj = np.zeros((max_size, max_size))

	for (a,b) in gtrain.edges():
		adj[a,b] = 1
		adj[b,a] = 1

	beta = np.linalg.eigh(adj)[0].max() * forgetting_factor_scale

	S = np.linalg.pinv(np.eye(adj.shape[0]) - beta*adj) - np.eye(adj.shape[0])

	ans = []

	if not ebunch is None:
		for (a,b) in test_edges_nonedges:
			ans.append((a,b,S[a,b]))
	else:
		for a in range(max_size):
			for b in range(max_size):
				if not (a,b) in gtrain.edges():
					ans.append((a,b,S[a,b]))
	ans.sort(reverse = True, key=lambda x:x[2])
	return ans