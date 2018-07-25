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
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple


class AdvancedEncoderDecoder(TfEncoderDecoder):
	def __init__(self, beam_width=4, **kwargs):
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

			(forward_output, backward_output), final_state = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=forward_cell,
				cell_bw=backward_cell,
				inputs=rnn_inputs,
				sequence_length=self.encoder_lengths,
				scope='BLSTM_' + str(i),
				dtype=tf.float32)

			rnn_inputs = tf.concat([forward_output, backward_output], axis=2)

		self.encoder_outputs = rnn_inputs
		self.encoder_final_state = final_state


	"""
	Modified to include LuongAttention (source below)
	
	Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
	"Effective Approaches to Attention-based Neural Machine Translation."
	EMNLP 2015. https://arxiv.org/abs/1508.04025
	"""
	# def decoding_training(self):
	# 	# attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
	# 	# cells = []
	# 	# for i in range(self.num_layers):                   
	# 	#     cell = tf.nn.rnn_cell.LSTMCell(
	# 	# 		self.hidden_dim, 
	# 	# 		initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=i),
	# 	# 		state_is_tuple=True)

	# 	#     cell = tf.contrib.rnn.AttentionCellWrapper(
	# 	#         cell, attn_length=40, state_is_tuple=True)

	# 	#     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
	# 	#     cells.append(cell)

	# 	# self.decoder_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
	# 	# decoder_initial_state = self.decoder_cell.zero_state(self.batch_size, tf.float32)

	# 	attention_mechanism = tf.contrib.seq2seq.LuongAttention(
	# 		num_units=self.hidden_dim,
	# 		memory=self.encoder_outputs,
	# 		memory_sequence_length=self.encoder_lengths,
	# 		dtype=tf.float32)

	# 	# Build RNN with depth num_layers:
	# 	self.decoder_cell = tf.contrib.rnn.MultiRNNCell([
	# 		tf.nn.rnn_cell.LSTMCell(
	# 		self.hidden_dim, 
	# 		initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)) for _ in range(self.num_layers)])

	# 	self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
	# 		cell=self.decoder_cell,
	# 		attention_mechanism=attention_mechanism,
	# 		attention_layer_size=self.hidden_dim)
	# 	decoder_initial_state = self.decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_final_state)

	# 	# self.decoder_cell = tf.contrib.rnn.MultiRNNCell([
	# 	# 	tf.nn.rnn_cell.LSTMCell(
	# 	# 	self.hidden_dim, 
	# 	# 	initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)) for _ in range(self.num_layers)])
		
	# 	# Add a dense layer (see imports):
	# 	# (Output indexes self.vocab)
	# 	self.output_layer = Dense(
	# 		units=self.vocab_size,
	# 		kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
		
	# 	# Helper for sampling:
	# 	training_helper = tf.contrib.seq2seq.TrainingHelper(
	# 		inputs=self.embedded_decoder_inputs,
	# 		sequence_length=self.decoder_lengths,
	# 		time_major=True)
		
	# 	# Dynamic decoding:
	# 	training_decoder = tf.contrib.seq2seq.BasicDecoder(
	# 		cell=self.decoder_cell,
	# 		helper=training_helper,
	# 		initial_state=decoder_initial_state,
	# 		output_layer=self.output_layer)
		
	# 	training_outputs = tf.contrib.seq2seq.dynamic_decode(
	# 		decoder=training_decoder,
	# 		impute_finished=True,
	# 		maximum_iterations=self.max_output_length)[0]
		
	# 	self.training_logits = training_outputs.rnn_output


	# """
	# Decoding inference with beam search
	# """
	def decoding_inference(self):
		start_tokens = tf.tile(
			input=tf.constant([self.vocab.index("<START>")], dtype=tf.int32),
			multiples=[self.batch_size])

		decoder_initial_state = tf.contrib.seq2seq.tile_batch(self.encoder_final_state, multiplier=self.beam_width)
		# decoder_initial_state = self.decoder_cell.zero_state(self.batch_size * self.beam_width, tf.float32).clone(cell_state=self.encoder_final_state)

		inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
			cell=self.decoder_cell,
			embedding=self.embedding_space,
			start_tokens=start_tokens,
			end_token=self.vocab.index("<END>"),
			initial_state=decoder_initial_state,
			beam_width=self.beam_width,
			output_layer=self.output_layer,
			length_penalty_weight=0.0)

		inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(
			inference_decoder,
			impute_finished=False,
			maximum_iterations=self.max_output_length)[0]

		self.inference_logits = inference_decoder_output.predicted_ids[:,:,0]
		self.inference_logits = tf.identity(self.inference_logits, name="inference_logits")


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

	seq2seq = AdvancedEncoderDecoder(
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
