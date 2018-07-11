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

train_iterations = 1500
learning_rate = 0.1
max_input_length = 6 # length of goals list
max_output_length = 10

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
		x_lengths = [len(seq) for seq in X] # len(seq) == 6
		num_examples = X.shape[0]
		length = X.shape[1]

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
			# train_data.append()

	seq2seq = Negotiator(
		vocab=parser.vocab,
		max_iter=1500,
		eta=0.1)