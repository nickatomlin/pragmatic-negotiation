import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import random
import json
from tensorflow.python.layers.core import Dense

import sys
sys.path.append("../")
sys.path.append('../../data/')
from agent import Agent
from parse import SentenceParser

train_iterations = 300
learning_rate = 0.1
max_length = 22
unk_threshold = 20


class HierarchicalAgent(Agent):
	def __init__(self, max_turns=15, **kwargs):
		self.max_turns = max_turns # Max number of dialogue turns (i.e., utterances)
		super(HierarchicalAgent, self).__init__(**kwargs)
	

	def _init_placeholders(self):
		self.encoder_inputs = tf.placeholder(
			shape=[self.max_turns, self.batch_size, self.max_input_length],
			dtype=tf.int32,
			name="encoder_inputs")

		self.encoder_lengths = tf.placeholder(
			shape=[self.max_turns, self.batch_size],
			dtype=tf.int32,
			name="encoder_lengths")

		self.decoder_inputs = tf.placeholder(
			shape=[self.max_turns, self.batch_size, self.max_output_length],
			dtype=tf.int32,
			name="decoder_inputs")

		self.decoder_targets = tf.placeholder(
			shape=[self.max_turns, self.batch_size, self.max_output_length],
			dtype=tf.int32,
			name="decoder_targets")

		self.decoder_lengths = tf.placeholder(
			shape=[self.max_turns, self.batch_size],
			dtype=tf.int32,
			name="decoder_lengths")
		

	def encoding_layer(self):
		self.encoder_cell = tf.contrib.rnn.MultiRNNCell([
			tf.nn.rnn_cell.LSTMCell(
				self.hidden_dim, 
				initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=1),
				reuse=tf.AUTO_REUSE)
			for _ in range(self.num_layers)])
		
		self.encoder_final_states = tf.map_fn(
			fn=self.encode_step, 
			elems=(self.embedded_encoder_inputs, self.encoder_lengths), dtype=tf.float32)


	def encode_step(self, args):
		embedded_encoder_inputs = args[0]
		encoder_lengths = args[1]
		
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
			cell=self.encoder_cell,
			inputs=embedded_encoder_inputs,
			sequence_length=encoder_lengths,
			dtype=tf.float32)
		
		return encoder_final_state[-1][1]
		

	def context_layer(self):
		self.context_cell = tf.contrib.rnn.MultiRNNCell([
			tf.nn.rnn_cell.LSTMCell(
				self.hidden_dim, 
				initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
			for _ in range(self.num_layers)])
		
		context_outputs, context_final_state = tf.nn.dynamic_rnn(
			cell=self.context_cell,
			inputs=self.encoder_final_states,
			sequence_length=[self.max_turns] * self.batch_size,
			dtype=tf.float32,
			scope="context_layer",
			time_major=True)
		
		self.context_outputs = context_outputs
		

	def decoding_layer(self):
		self.decoder_cell = tf.contrib.rnn.MultiRNNCell([
			tf.nn.rnn_cell.GRUCell(self.hidden_dim) for _ in range(self.num_layers)])
		
		self.output_layer = Dense(
			units=self.vocab_size,
			kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
		
		self.decoding_training()
		self.decoding_inference()
	

	def decoding_training(self):
		ta = tf.TensorArray(dtype=tf.float32, size=self.max_turns)
		
		_, training_logits = tf.while_loop(
			cond=lambda i,t: i < self.max_turns,
			body=self.decoding_training_step,
			loop_vars=[0, ta])
		
		self.training_logits = training_logits.stack()
		

	def decoding_training_step(self, i, ta):
		context_state = tf.gather(self.context_outputs, i)
		embedded_decoder_inputs = tf.gather(self.embedded_decoder_inputs, i)
		decoder_lengths = tf.gather(self.decoder_lengths, i)
		
		training_helper = tf.contrib.seq2seq.TrainingHelper(
			inputs=embedded_decoder_inputs,
			sequence_length=decoder_lengths,
			time_major=False)
		
		training_decoder = tf.contrib.seq2seq.BasicDecoder(
			cell=self.decoder_cell,
			helper=training_helper,
			initial_state=(context_state,context_state),
			output_layer=self.output_layer)

		training_outputs = tf.contrib.seq2seq.dynamic_decode(
			decoder=training_decoder,
			impute_finished=True,
			maximum_iterations=self.max_output_length)[0]
				
		return (i+1, ta.write(i, training_outputs.rnn_output))
	

	def decoding_inference(self):
		ta = tf.TensorArray(dtype=tf.int32, size=self.max_turns)
		
		_, inference_logits = tf.while_loop(
			cond=lambda i,t: i < self.max_turns,
			body=self.decoding_inference_step,
			loop_vars=[0, ta])
		
		self.inference_logits = inference_logits.stack()
		self.inference_logits = tf.identity(self.inference_logits, name="inference_logits")
	

	def decoding_inference_step(self, i, ta):
		context_state = tf.gather(self.context_outputs, i)
		
		start_tokens = tf.tile(
			input=tf.constant([self.vocab.index("<START>")], dtype=tf.int32),
			multiples=[self.batch_size])

		inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
			embedding=self.embedding_space,
			start_tokens=start_tokens,
			end_token=self.vocab.index("<END>"))

		inference_decoder = tf.contrib.seq2seq.BasicDecoder(
			cell=self.decoder_cell,
			helper=inference_helper,
			initial_state=(context_state, context_state),
			output_layer=self.output_layer)

		inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(
			inference_decoder,
			impute_finished=True,
			maximum_iterations=self.max_output_length)[0]
		
		inference_logits = inference_decoder_output.sample_id
		return (i+1, ta.write(i, inference_logits))
	

	def build_graph(self):
		self._init_placeholders()
		self._define_embedding()
		self.encoding_layer()
		self.context_layer()
		self.decoding_layer()
		

	def train_dict(self, X, y):
		decoder_inputs = [["<START> " + sent for sent in dialogue] for dialogue in y]
		decoder_targets = [[sent + " <END>" for sent in dialogue] for dialogue in y]
		
		encoder_inputs, encoder_lengths = self.prepare_data(X)
		decoder_inputs, _ = self.prepare_data(y)
		decoder_targets, decoder_lengths = self.prepare_data(decoder_targets)
		
		return {self.encoder_inputs: encoder_inputs,
			self.decoder_inputs: decoder_inputs,
			self.decoder_targets: decoder_targets,
			self.encoder_lengths: encoder_lengths,
			self.decoder_lengths: decoder_lengths}
	

	def prepare_data(self, data):
		batch_size = self.batch_size
		max_length = self.max_input_length
		
		index = dict(zip(self.vocab, range(len(self.vocab))))
		unk_index = index['$UNK']
		
		new_data = np.zeros((self.max_turns, batch_size, max_length), dtype='int')
		ex_lengths = np.zeros((self.max_turns, batch_size))
		max_num_turns = 0
		for batch in range(len(data)):
			num_turns = len(data[batch])
			if num_turns > max_num_turns:
				max_num_turns = num_turns
			for turn in range(min(num_turns, self.max_turns)):
				ex_lengths[turn][batch] = self.max_input_length
				vals = data[batch][turn][-max_length: ].split()
				vals = [index.get(w, unk_index) for w in vals]
				temp = np.zeros((max_length,), dtype='int')
				temp[0: len(vals)] = vals
				new_data[turn][batch] = temp
			
		return new_data, ex_lengths
	

	def get_cost_function(self, **kwargs):
		seq_length = self.max_turns*self.max_output_length
		
		training_logits = tf.transpose(self.training_logits, [1, 0, 2, 3])
		training_logits = tf.reshape(training_logits, [self.batch_size, seq_length, self.vocab_size])
		
		decoder_targets = tf.transpose(self.decoder_targets, [1, 0, 2])
		decoder_targets = tf.reshape(decoder_targets, [self.batch_size, seq_length])
		
		decoder_lengths = tf.reduce_sum(self.decoder_lengths, 0)
		masks = tf.sequence_mask(decoder_lengths, seq_length, dtype=tf.float32, name='masks')
		cost = tf.contrib.seq2seq.sequence_loss(
			logits=training_logits,
			targets=decoder_targets,
			weights=masks)
		return cost


if __name__ == '__main__':
	simple_example()
