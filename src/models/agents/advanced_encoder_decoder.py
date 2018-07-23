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
		cells_fw = []
		cells_bw = []
		for _ in range(self.num_layers):


		# Build RNN with depth num_layers:	
		encoder_cell = tf.contrib.rnn.MultiRNNCell([
			tf.nn.rnn_cell.LSTMCell(
				self.hidden_dim, 
				initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
			for _ in range(self.num_layers)])
		
		# Run the RNN:
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
			cell=encoder_cell,
			inputs=self.embedded_encoder_inputs,
			sequence_length=self.encoder_lengths,
			dtype=tf.float32,
			scope="encoding_layer")
		
		self.encoder_final_state = encoder_final_state