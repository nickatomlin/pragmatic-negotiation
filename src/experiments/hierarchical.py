"""
Imports
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
from hierarchical_agent import HierarchicalAgent
from parse import SentenceParser

"""
Parameters
"""
train_iterations = 300
learning_rate = 0.1
max_length = 22
unk_threshold = 20

"""
Parsing
"""
parser = SentenceParser(unk_threshold=unk_threshold,
                  input_directory="../../data/raw/",
                  output_directory="../../data/tmp/")
print("Vocab size: {}".format(parser.vocab_size))
parser.parse()


"""
Agent Initialization
"""
tf.reset_default_graph()
agent = HierarchicalAgent(vocab=parser.vocab,
    max_iter=train_iterations,
	eta=learning_rate,
	max_input_length=max_length,
	max_output_length=max_length,
	hidden_dim=64)

"""
Data Parsing
"""
train_data = []
with open("../../data/tmp/train.txt", "r") as train_file:
    for line in train_file:
        train_example = json.loads(line)
        train_data.append(([""] + train_example[:-1], train_example))

X, y = zip(*train_data)

"""
Training
"""
agent.fit(X, y)
