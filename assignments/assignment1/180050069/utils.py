import networkx as nx
import numpy as np

def find_nbr_nonnbr(G):
	"""
	A routine that processes a networkx graph and emits list of neighbours and non-neighbours for each node.
	Input: NetworkX graph
	Returns: dictionary of neighbour and non-neighbors
	Do not use on large graphs since non-neighbour dictionary is O(n^3) storage, n: number of vertices. 
	"""
	
	vertex_set  = set(G.nodes())
	vertex_list = list(vertex_set)
	
	nbr_dict, nonnbr_dict = {}, {}

	for node in range(len(vertex_list)):
		nbr_set = set([nbr for nbr in G[node]])
		nonnbr_set = list(vertex_set - nbr_set)

		nbr_dict[node] = nbr_set
		nonnbr_dict[node] = nonnbr_set

	return nbr_dict, nonnbr_dict

def MAP(index_list, labels, tot_vetices, topk = None):
	ans = np.zeros((tot_vetices,))
	count = np.zeros((tot_vetices,))
	corr = np.zeros((tot_vetices,))

	index_list.sort(reverse = True, key=lambda x:x[2])
	for i,(a,b,_) in enumerate(index_list):
		if not topk is None:
			if i > topk:
				break
		if a > b:
			temp = a
			a = b
			b = temp
		count[a] += 1
		count[b] += 1
		if labels[(a,b)] == 1:
			corr[a] += 1
			corr[b] += 1

			ans[a] += corr[a]/count[a]
			ans[b] += corr[b]/count[b]

	corr_indices = corr > 0
	out = np.divide(ans[corr_indices],corr[corr_indices])
	return out.mean()

def MRR(index_list, labels, tot_vetices, topk = None):
	ans = np.zeros((tot_vetices,))-1
	count = np.zeros((tot_vetices,))
	index_list.sort(reverse = True, key=lambda x:x[2])
	for i,(a,b,_) in enumerate(index_list):
		if not topk is None:
			if i > topk:
				break
		if a > b:
			temp = a
			a = b
			b = temp
		count[a] += 1
		count[b] += 1
		if labels[(a,b)] == 1:
			if ans[a] == -1:
				ans[a] = 1/count[a]
			if ans[b] == -1:
				ans[b] = 1/count[b]
	ans_indices = np.invert(ans == -1)
	count_indices = np.invert(count == 0)
	return ans[ans_indices].sum()/ans_indices.sum()

def common_neighbor(gtrain, ebunch = None):
	ans = []
	if not ebunch is None:
		for (a,b) in ebunch:
			# b_neighbors = set(gtrain.neighbors(b))
			# a_neighbors = set(gtrain.neighbors(a))
			# cn = len(set.intersection(a_neighbors, b_neighbors))
			cn = len(list(nx.common_neighbors(gtrain, a, b)))
			ans.append((a,b,cn))
	else:
		for a in range(max_size):
			for b in range(max_size):
				if not (a,b) in gtrain.edges():
					b_neighbors = set(gtrain.neighbors(b))
					a_neighbors = set(gtrain.neighbors(a))
					cn = len(set.intersection(a_neighbors, b_neighbors))
					ans.append((a,b,cn))
	ans.sort(reverse = True, key=lambda x:x[2])
	return ans


def katz_measure(gtrain, forgetting_factor_scale = 0.5, ebunch = None):
	max_size = max(gtrain.nodes())+1
	adj = np.zeros((max_size, max_size))

	for (a,b) in gtrain.edges():
		if a > b:
			continue
		adj[a,b] = 1
		adj[b,a] = 1

	beta = (1/np.linalg.eigh(adj)[0].max()) * forgetting_factor_scale

	S = np.linalg.pinv(np.eye(adj.shape[0]) - beta*adj) - np.eye(adj.shape[0])

	ans = []

	if not ebunch is None:
		for (a,b) in ebunch:
			ans.append((a,b,S[a,b]))
	else:
		for a in range(max_size):
			for b in range(max_size):
				if not (a,b) in gtrain.edges():
					ans.append((a,b,S[a,b]))
	ans.sort(reverse = True, key=lambda x:x[2])
	return ans
