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

class AdvancedEncoderDecoder(TfEncoderDecoder):
	"""
	Encoding layer with bidirectional RNNs
	"""
	def encoding_layer(self):

		rnn_inputs = self.embedded_encoder_inputs
		for i in range(self.num_layers):
			forward_cell = tf.nn.rnn_cell.LSTMCell(
				self.hidden_dim, 
				initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2*i)
			backward_cell = tf.nn.rnn_cell.LSTMCell(
				self.hidden_dim, 
				initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=(2*i+1))

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