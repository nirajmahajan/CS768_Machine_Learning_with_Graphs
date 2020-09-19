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

with open('./pickles/graph.pickle', 'rb') as handle:
	graph = pickle.load(handle)['graph']

with open('datasets/facebookmini_eq.txt', 'w') as f:
	for a in range(500):
		for b in range(500):
			if random.randint(1,2) == 1:
				continue
			else:
				f.write("{} {}\n".format(a,b))
	