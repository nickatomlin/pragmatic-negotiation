"""
Baseline seq2seq tests for negotiation.
"""
import json
import numpy as np
import tensorflow as tf

# Import seq2seq model and parser:
import sys
sys.path.append('../models/')
sys.path.append('../data/')
from tf_encoder_decoder import TfEncoderDecoder
from parse import FBParser

train_iterations = 2500
learning_rate = 0.1
max_input_length = 6 # length of goals list
max_output_length = 10

results_file = "results/baseline.txt"

# Create a new class that modifies our baselines seq2seq
# -> Necessary for modifying input data
class Negotiator(TfEncoderDecoder):
	# Don't call prepare_data() on X:
	def train_dict(self, X, y):
		encoder_inputs = X
		encoder_lengths = [len(seq) for seq in X] # len(seq) == 6

		decoder_inputs = [["<START>"] + list(seq) for seq in y]
		decoder_targets = [list(seq) + ["<END>"] for seq in y]
		decoder_inputs, _ = self.prepare_data(decoder_inputs, self.max_output_length)
		decoder_targets, decoder_lengths = self.prepare_data(decoder_targets, self.max_output_length)

		return {self.encoder_inputs: encoder_inputs,
				self.decoder_inputs: decoder_inputs,
				self.decoder_targets: decoder_targets,
				self.encoder_lengths: encoder_lengths,
				self.decoder_lengths: decoder_lengths}

	def predict(self, X):
		X = np.asarray(list(X))
		x_lengths = [len(seq) for seq in X] # len(seq) == 6
		num_examples = len(X)
		length = 6

		# Resize X and x_lengths to match the size of inference_logits:
		X.resize((self.batch_size, length))
		x_lengths = np.asarray(x_lengths)
		x_lengths.resize(self.batch_size)

		answer_logits = self.sess.run(self.inference_logits, {
			self.encoder_inputs: X, 
			self.encoder_lengths: x_lengths})[:num_examples]
		return answer_logits


if __name__ == '__main__':
	# Setup parser:
	parser = FBParser()
	print("Vocab size: {}".format(parser.vocab_size))

	# Load processed data:
	train_data = []
	test_data = []

	with open("../../data/processed/train.txt", "r") as train_file, open("../../data/processed/test.txt", "r") as test_file:
		for line in train_file:
			train_example = json.loads(line)
			train_data.append((
				train_example["input"],
				train_example["output"][0]))

		for line in test_file:
			test_example = json.loads(line)
			test_data.append((
				test_example["input"],
				test_example["output"][0]))

	X, y = zip(*train_data)
	X_test, y_test = zip(*test_data[:20])

	test_inputs = X_test
	test_strings = [''.join(seq) for seq in y_test]

	with open(results_file, "w") as f:
		f.write('\nTest inputs:\n')
		f.write(str(test_inputs))
		f.write('\nTest strings:\n')
		f.write(str(test_strings))

		seq2seq = Negotiator(
			vocab=parser.vocab,
			max_iter=train_iterations,
			eta=learning_rate,
			max_input_length=max_input_length,
			max_output_length=max_output_length,
			hidden_dim=64)

		seq2seq.fit(X, y)
		logits = seq2seq.predict(X_test)

		f.write('\nPredictions:\n')
		f.write(str(seq2seq.output(logits, padding=" ")))
