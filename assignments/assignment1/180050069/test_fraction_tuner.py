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

if args.test_fraction == 0.1:
	for tfi in range(60,70,1):
		tf = tfi/1000
		print('Using Test Fraction (for code) = {}\n'.format(tf))
		# Calculate observed test_fraction
		graph = Graph(filename)
		# graph.split_train_test(test_fraction_mapping[args.test_fraction])
		graph.split_train_test(tf)

		print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
		print ("Observed fraction of train non edges	  : %0.4f" % (len(graph.G_train_invert.edges())/(len(graph.G_train_invert.edges()) + len(graph.G_test_invert.edges()))))
	os._exit(0)


if args.test_fraction == 0.2:
	for tfi in range(110,120,1):
		tf = tfi/1000
		print('Using Test Fraction (for code) = {}\n'.format(tf))
		# Calculate observed test_fraction
		graph = Graph(filename)
		# graph.split_train_test(test_fraction_mapping[args.test_fraction])
		graph.split_train_test(tf)

		print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
		print ("Observed fraction of train non edges	  : %0.4f" % (len(graph.G_train_invert.edges())/(len(graph.G_train_invert.edges()) + len(graph.G_test_invert.edges()))))
	os._exit(0)

if args.test_fraction == 0.3:
	for tfi in range(170,180,1):
		tf = tfi/1000
		print('Using Test Fraction (for code) = {}\n'.format(tf))
		# Calculate observed test_fraction
		graph = Graph(filename)
		# graph.split_train_test(test_fraction_mapping[args.test_fraction])
		graph.split_train_test(tf)

		print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
		print ("Observed fraction of train non edges	  : %0.4f" % (len(graph.G_train_invert.edges())/(len(graph.G_train_invert.edges()) + len(graph.G_test_invert.edges()))))
	os._exit(0)

if args.test_fraction == 0.4:
	for tfi in range(230,240,1):
		tf = tfi/1000
		print('Using Test Fraction (for code) = {}\n'.format(tf))
		# Calculate observed test_fraction
		graph = Graph(filename)
		# graph.split_train_test(test_fraction_mapping[args.test_fraction])
		graph.split_train_test(tf)

		print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
		print ("Observed fraction of train non edges	  : %0.4f" % (len(graph.G_train_invert.edges())/(len(graph.G_train_invert.edges()) + len(graph.G_test_invert.edges()))))
	os._exit(0)

if args.test_fraction == 0:
	for tfi in range(5,10,1):
		tf = tfi/100
		print('Using Test Fraction (for code) = {}\n'.format(tf))
		# Calculate observed test_fraction
		graph = Graph(filename)
		# graph.split_train_test(test_fraction_mapping[args.test_fraction])
		graph.split_train_test(tf)

		print ("Observed fraction of train edges	  : %0.4f" % (len(graph.G_train.edges())/(len(graph.G.edges()))))
		print ("Observed fraction of train non edges	  : %0.4f" % (len(graph.G_train_invert.edges())/(len(graph.G_train_invert.edges()) + len(graph.G_test_invert.edges()))))