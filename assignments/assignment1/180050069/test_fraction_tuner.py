import os
import copy
import random
import argparse
import networkx as nx

from random import sample
from classes import Graph

# constant declarations
filename = 'datasets/facebook.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--test_fraction', type = float, default = 0)
args = parser.parse_args()

random.seed(0)

G = nx.read_edgelist(filename)
graph = Graph(G)


if args.test_fraction == 0.1:
	for tfi in range(1):
		tf = 0.11
		print('\nUsing Test Fraction (for code) = {}'.format(tf))
		# Calculate observed test_fraction
		# graph.split_train_test(test_fraction_mapping[args.test_fraction])
		random.seed(0)
		graph.split_train_test(tf)

		print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
	os._exit(0)


if args.test_fraction == 0.2:
	for tfi in range(210,220,1):
		tf = tfi/1000
		print('\nUsing Test Fraction (for code) = {}'.format(tf))
		# Calculate observed test_fraction
		# graph.split_train_test(test_fraction_mapping[args.test_fraction])
		random.seed(0)
		graph.split_train_test(tf)

		print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
	os._exit(0)

if args.test_fraction == 0.3:
	for tfi in range(300,310,1):
		tf = tfi/1000
		print('\nUsing Test Fraction (for code) = {}'.format(tf))
		# Calculate observed test_fraction
		# graph.split_train_test(test_fraction_mapping[args.test_fraction])
		random.seed(0)
		graph.split_train_test(tf)

		print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
	os._exit(0)

if args.test_fraction == 0.4:
	for tfi in range(410,420,1):
		tf = tfi/1000
		print('\nUsing Test Fraction (for code) = {}'.format(tf))
		# Calculate observed test_fraction
		# graph.split_train_test(test_fraction_mapping[args.test_fraction])
		random.seed(0)
		graph.split_train_test(tf)

		print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
	os._exit(0)

if args.test_fraction == 0:
	for tfi in range(10,50,1):
		tf = tfi/100
		print('\nUsing Test Fraction (for code) = {}'.format(tf))
		# Calculate observed test_fraction
		# graph.split_train_test(test_fraction_mapping[args.test_fraction])
		random.seed(0)
		graph.split_train_test(tf)

		print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
	os._exit(0)