"""
Baseline seq2seq tests for negotiation.
"""
import json
import numpy as np
import pandas as pd
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
max_output_length = 20
unk_threshold = 20

results_file = "results/baseline2.txt"

# Create a new class that modifies our baselines seq2seq
# -> Necessary for modifying input data
class Negotiator(TfEncoderDecoder):
	# Don't call prepare_data() on X:
	def fit(self, X, y, **kwargs):
		"""
		Key modifications:
		 - Soft placement of CPU/GPU devices
		 - Ability to save model
		"""
		if isinstance(X, pd.DataFrame):
			X = X.values

		# Incremental performance:
		X_dev = kwargs.get('X_dev')
		if X_dev is not None:
			dev_iter = kwargs.get('test_iter', 10)

		# One-hot encoding of target `y`, and creation
		# of a class attribute.
		y = self.prepare_output_data(y)

		self.input_dim = len(X[0])

		# Start the session:
		tf.reset_default_graph()
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

		# Build the computation graph. This method is instantiated by
		# individual subclasses. It defines the model.
		self.build_graph()

		# Optimizer set-up:
		self.cost = self.get_cost_function()
		self.optimizer = self.get_optimizer()

		# Initialize the session variables:
		self.sess.run(tf.global_variables_initializer())

		# Training, full dataset for each iteration:
		for i in range(1, self.max_iter+1):
			loss = 0
			for X_batch, y_batch in self.batch_iterator(X, y):
				_, batch_loss = self.sess.run(
					[self.optimizer, self.cost],
					feed_dict=self.train_dict(X_batch, y_batch))
				loss += batch_loss
			self.errors.append(loss)
			if X_dev is not None and i > 0 and i % dev_iter == 0:
				self.dev_predictions.append(self.predict(X_dev))
			if loss < self.tol:
				self._progressbar("stopping with loss < self.tol", i)
				break
			else:
				self._progressbar("loss: {}".format(loss), i)

		return self

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
	parser = FBParser(unk_threshold=unk_threshold)
	print("Vocab size: {}".format(parser.vocab_size))

	# Load processed data:
	train_data = []
	test_data = []

	with open("../../data/processed/train.txt", "r") as train_file, open("../../data/processed/test.txt", "r") as test_file:
		for line in train_file:
			train_example = json.loads(line)
			train_data.append((
				train_example["input"],
				train_example["output"][0].split()))

		for line in test_file:
			test_example = json.loads(line)
			test_data.append((
				test_example["input"],
				test_example["output"][0].split()))

	X, y = zip(*train_data)
	X_test, y_test = zip(*test_data[:20])

	test_inputs = X_test
	test_strings = [' '.join(seq) for seq in y_test]

	with open(results_file, "w") as f:
		# Save model file:
		# saver = tf.train.Saver()

		f.write('\nTest inputs:\n')
		for seq in test_inputs:
			f.write(str(seq))
			f.write("\n")

		f.write('\nTest strings:\n')
		for string in test_strings:
			f.write(string)
			f.write("\n")
		# f.write(str(test_inputs))
		# f.write(str((test_strings)))

		seq2seq = Negotiator(
			vocab=parser.vocab,
			max_iter=train_iterations,
			eta=learning_rate,
			max_input_length=max_input_length,
			max_output_length=max_output_length,
			hidden_dim=64)

		seq2seq.fit(X, y)
		# save_path = saver.save(seq2seq.sess, "../../models/seq2seq_baseline.ckpt")
		# print("Model saved in path: %s" % save_path)

		logits = seq2seq.predict(X_test)
		f.write('\nPredictions:\n')
		for string in seq2seq.output(logits, padding=" "):
			f.write(string)
			f.write("\n")
		# f.write(str("".join(seq2seq.output(logits, padding=" "))))
