"""
Modification of the baseline seq2seq model with the following features:
 - Bidirectional RNNs
 - Attention
 - Beam search
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf

import sys
sys.path.append("..")
from tf_encoder_decoder import TfEncoderDecoder

import random
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense

class AdvancedEncoderDecoder(TfEncoderDecoder):
	def __init__(self, beam_width):
		self.beam_width = beam_width
		super(AdvancedEncoderDecoder, self).__init__(**kwargs)

	"""
	Encoding layer with bidirectional RNNs
	"""
	def encoding_layer(self):
		rnn_inputs = self.embedded_encoder_inputs
		for i in range(self.num_layers):
			forward_cell = tf.nn.rnn_cell.LSTMCell(
				self.hidden_dim, 
				initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2*i))
			backward_cell = tf.nn.rnn_cell.LSTMCell(
				self.hidden_dim, 
				initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=(2*i+1)))

			forward_state = forward_cell.zero_state(self.batch_size, tf.float32)
			backward_state = backward_cell.zero_state(self.batch_size, tf.float32)

			(forward_output, backward_output) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=forward_cell,
				cell_bw=backward_cell,
				inputs=rnn_inputs,
				initial_state_fw=forward_state,
				initial_state_bw=backward_state,
				sequence_length=self.encoder_lengths,
				time_major=True)

			rnn_inputs = tf.concat([forward_output, backward_output], axis=2)

		self.encoder_final_state = rnn_inputs


	def decoding_training(self):
		# Build RNN with depth num_layers:
		self.decoder_cell = tf.contrib.rnn.MultiRNNCell([
			tf.nn.rnn_cell.LSTMCell(
			self.hidden_dim, 
			initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)) for _ in range(self.num_layers)])
		
		# Add a dense layer (see imports):
		# (Output indexes self.vocab)
		self.output_layer = Dense(
			units=self.vocab_size,
			kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
		
		# Helper for sampling:
		training_helper = tf.contrib.seq2seq.TrainingHelper(
			inputs=self.embedded_decoder_inputs,
			sequence_length=self.decoder_lengths,
			time_major=False)
		
		# Dynamic decoding:
		training_decoder = tf.contrib.seq2seq.BasicDecoder(
			cell=self.decoder_cell,
			helper=training_helper,
			initial_state=self.encoder_final_state,
			output_layer=self.output_layer)
		
		training_outputs = tf.contrib.seq2seq.dynamic_decode(
			decoder=training_decoder,
			impute_finished=True,
			maximum_iterations=self.max_output_length)[0]
		
		self.training_logits = training_outputs.rnn_output


	"""
	Decoding inference with beam search
	"""
	def decoding_inference(self):
		start_tokens = tf.tile(
			input=tf.constant([self.vocab.index("<START>")], dtype=tf.int32),
			multiples=[self.batch_size])

		decoder_initial_state = tf.contrib.seq2seq.tile_batch(
    		t=self.encoder_final_state,
    		multiplier=self.beam_width)

		inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
	        cell=self.decoder_cell,
	        embedding=self.embedding_space,
	        start_tokens=start_tokens,
	        end_token=self.vocab.index("<END>"),
	        initial_state=decoder_initial_state,
	        beam_width=self.beam_width,
	        length_penalty_weight=0.0)

		inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(
			inference_decoder,
			impute_finished=True,
			maximum_iterations=self.max_output_length)[0]

		self.inference_logits = inference_decoder_output.sample_id


def simple_example():
	vocab = ['<PAD>', '$UNK', '<START>', '<END>', 'a', 'b']

	data = []
	for i in range(100):
		input_string = ""
		output_string = ""
		length = random.randint(1,7)
		for char in range(length):
			if (random.random() > 0.5):
				input_string += "a"
				output_string += "b"
			else:
				input_string += "b"
				output_string += "a"
			data.append([np.asarray(list(input_string)), np.asarray(list(output_string))])

	# Modify test_size to increase the number of test examples:
	train, test = train_test_split(data, test_size=4)

	seq2seq = TfEncoderDecoder(
		vocab=vocab, max_iter=1500, eta=0.1)

	X, y = zip(*train)
	seq2seq.fit(X, y)

	X_test, _ = zip(*test)
	logits = seq2seq.predict(X_test)

	test_strings = [''.join(seq) for seq in X_test]

	print('\nTest data:', test_strings)
	print('Predictions:', seq2seq.output(logits))

if __name__ == '__main__':
	simple_example()
