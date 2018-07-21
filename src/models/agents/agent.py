"""
Baseline dialogue agent for seq2seq.

Encoder inputs: value state (list of 6 tokens)
Decoder inputs/targets: list of word indices
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf

import sys
sys.path.append("..")
from tf_encoder_decoder import TfEncoderDecoder

class Agent(TfEncoderDecoder):
	def fit(self, X, y, save_path="../../../models/seq2seq", **kwargs):
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

		# Save the model:
		saver = tf.train.Saver()

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

		path = saver.save(self.sess, save_path)
		print("Model saved in path: %s" % path)
		return self


	def train_dict(self, X, y):
		"""
		Key modifications:
		 - Don't call prepare_data() on X
		"""
		encoder_inputs = X
		encoder_lengths = [len(seq) for seq in X] # len(seq) == 6

		decoder_inputs = [["<eos>"] + list(seq) for seq in y]
		decoder_targets = [list(seq) + ["<END>"] for seq in y]
		decoder_inputs, _ = self.prepare_data(decoder_inputs, self.max_output_length)
		decoder_targets, decoder_lengths = self.prepare_data(decoder_targets, self.max_output_length)

		return {self.encoder_inputs: encoder_inputs,
				self.decoder_inputs: decoder_inputs,
				self.decoder_targets: decoder_targets,
				self.encoder_lengths: encoder_lengths,
				self.decoder_lengths: decoder_lengths}

	def predict(self, X):
		"""
		Key modifications:
		 - Don't call prepare_data() on X
		 - Alternate calculation of num_examples, length
		"""
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
		

	def decoding_inference(self):
		start_tokens = tf.tile(
			input=tf.constant([self.vocab.index("<eos>")], dtype=tf.int32),
			multiples=[self.batch_size])

		inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
			embedding=self.embedding_space,
			start_tokens=start_tokens,
			end_token=self.vocab.index("<END>"))

		inference_decoder = tf.contrib.seq2seq.BasicDecoder(
			cell=self.decoder_cell,
			helper=inference_helper,
			initial_state=self.encoder_final_state,
			output_layer=self.output_layer)

		inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(
			inference_decoder,
			impute_finished=True,
			maximum_iterations=self.max_output_length)[0]

		self.inference_logits = inference_decoder_output.sample_id
		self.inference_logits = tf.identity(self.inference_logits, name="inference_logits")