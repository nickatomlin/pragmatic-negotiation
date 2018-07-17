"""
Testing:
 - Model saving/loading
 - Responding to dialogue
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import random
import json
from tensorflow.python.layers.core import Dense

import sys
sys.path.append("../models/")
sys.path.append('../models/agents/')
sys.path.append('../data/')
from agent import Agent
from parse import FBParser

train_iterations = 2500
learning_rate = 0.1
max_input_length = 6 # length of goals list
max_output_length = 20
unk_threshold = 20

if __name__ == '__main__':
	# Build FBParser and Agent:
	parser = FBParser(
		unk_threshold=unk_threshold,
		input_directory="../../data/raw/",
        output_directory="../../data/processed/")
	print("Vocab size: {}".format(parser.vocab_size))

	agent = Agent(vocab=parser.vocab,
              max_iter=train_iterations,
              eta=learning_rate,
              max_input_length=max_input_length,
              max_output_length=max_output_length,
              hidden_dim=64)

	# Training:
	train_data = []
	with open("../../data/processed/train.txt", "r") as train_file:
	    for line in train_file:
	        train_example = json.loads(line)
	        train_data.append((
	            train_example["input"],
	            train_example["output"][0].split()))

	X, y = zip(*train_data)
	agent.fit(X, y, save_path="../../models/seq2seq500")

	# Testing:
	tf.reset_default_graph()

	new_agent = Agent(vocab=parser.vocab,
	              max_iter=train_iterations,
	              eta=learning_rate,
	              max_input_length=max_input_length,
	              max_output_length=max_output_length,
	              hidden_dim=64)

	new_agent.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
	new_agent.build_graph()
	new_agent.sess.run(tf.global_variables_initializer())

	saver = tf.train.import_meta_graph('../../models/seq2seq500.meta')
	saver.restore(new_agent.sess, '../../models/seq2seq500')

	test_data = []
	with open("../../data/processed/test.txt", "r") as test_file:
	    for line in test_file:
	        test_example = json.loads(line)
	        test_data.append((
	            test_example["input"],
	            test_example["output"][0].split()))

	X_test, y_test = zip(*test_data[:20])
	print(X_test)
	logits = new_agent.predict(X_test)
	print('\nPredictions:\n')
	for string in new_agent.output(logits, padding=" "):
	    print(string)
	    print("\n")
