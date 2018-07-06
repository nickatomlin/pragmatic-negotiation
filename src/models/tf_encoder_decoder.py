import numpy as np
import tensorflow as tf
from tf_rnn_classifier import TfRNNClassifier
import warnings

__author__ = 'Nicholas Tomlin'

class TfEncoderDecoder(TfRNNClassifier):
	'''
	Parameters
	----------
	max_input_length : int
		TODO: Maximum sequence length for the input.
	max_output_length : int
		TODO: Maximum sequence length for the output. 
	vocab : list
		The full vocabulary. `_convert_X` will convert the data provided
		to `fit` and `predict` methods into a list of indices into this
		list of items. For now, assume the input and output have the
		same vocabulary.
	'''

	def __init__(self,
		max_input_length=5,
		max_output_length=5,
		**kwargs):
		self.max_input_length = max_input_length
		self.max_output_length = max_output_length
		super(TfEncoderDecoder, self).__init__(**kwargs)


	def build_graph(self):
		self._init_placeholders()
		self._define_embedding()

		self.encoding_layer()
		self.decoding_layer()


	def _init_placeholders(self):
		"""
		Helper function for build_graph which initializes seq2seq
		placeholders for encoder inputs and decoder targets
		"""
		self.encoder_inputs = tf.placeholder(
			shape=[None, None],
			dtype=tf.int32,
			name="encoder_inputs")

		self.encoder_lengths = tf.placeholder(
			shape=[None],
			dtype=tf.int32,
			name="encoder_lengths")

		self.decoder_inputs= tf.placeholder(
			shape=[None, None],
			dtype=tf.int32,
			name="decoder_inputs")

		self.decoder_targets = tf.placeholder(
			shape=[None, None],
			dtype=tf.int32,
			name="decoder_targets")

		self.decoder_lengths = tf.placeholder(
			shape=[None],
			dtype=tf.int32,
			name="decoder_lengths")


	def _define_embedding(self):
		"""Build the embedding matrix. If the user supplied a matrix, it
		is converted into a Tensor, else a random Tensor is built. This
		method sets `self.embedding` for use and returns None.
		"""
		self.embedding_encoder = tf.Variable(tf.random_uniform(
			shape=[self.vocab_size, self.embed_dim],
			minval=-1.0,
			maxval=1.0,
			name="embedding_encoder"))
		self.embedded_encoder_inputs = tf.nn.embedding_lookup(self.embedding_encoder, self.encoder_inputs)

		self.embedding_decoder = tf.Variable(tf.random_uniform(
			shape=[self.vocab_size, self.embed_dim],
			minval=-1.0,
			maxval=1.0,
			name="embedding_decoder"))
		self.embedded_decoder_inputs = tf.nn.embedding_lookup(self.embedding_decoder, self.decoder_inputs)


	def encoding_layer(self):
		# Build encoder RNN cell:
		encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, activation=self.hidden_activation)

		# Run the RNN:
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
			cell=encoder_cell,
			inputs=self.embedded_encoder_inputs,
			time_major=True,
			dtype=tf.float32,
			scope="encoding_layer")

		self.encoder_final_state = encoder_final_state


	def decoding_layer(self):
		# Build decoder RNN cell:
		decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, activation=self.hidden_activation)

		# Run the RNN:
		decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
		    decoder_cell,
		    self.embedded_decoder_inputs,
			initial_state=self.encoder_final_state,
			time_major=True,
			dtype=tf.float32,
			scope="decoding_layer")

		decoder_logits = tf.contrib.layers.linear(decoder_outputs, self.vocab_size)
		
		self.outputs = decoder_outputs
		self.model = decoder_logits


	def prepare_output_data(self, y):
		"""
		Modified to treat y as a sequence. Avoids one-hot
		encoding of y in fit(). Use _convert_X() instead.
		"""
		return y


	def get_cost_function(self, **kwargs):
		"""Uses `softmax_cross_entropy_with_logits` so the
		input should *not* have a softmax activation
		applied to it.
		"""
		return tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(
				logits=self.model,
				labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32)))


	def predict(self, X):
		"""
		TODO: test this. Still unsure what the decoder_inputs
		should look like. Should probably rename _convert_X().
		"""
		decoder_prediction = tf.argmax(self.model, 2)
		decoder_inputs = np.zeros_like(X)

		X, x_lengths = self._convert_X(X)
		y, y_lengths = self._convert_X(decoder_inputs)

		predictions = self.sess.run(
			decoder_prediction,
			feed_dict={
				self.encoder_inputs: X,
				self.encoder_lengths: x_lengths,
				self.decoder_targets: y,
				self.decoder_lengths: y_lengths
			})

		return predictions


	def train_dict(self, X, y):
		decoder_inputs = [["<eos>"] + list(seq) for seq in y]
		decoder_targets = [list(seq) + ["<eos>"] for seq in y]

		encoder_inputs, encoder_lengths = self._convert_X(X)
		decoder_inputs, decoder_lengths = self._convert_X(decoder_inputs)
		decoder_targets, _ = self._convert_X(decoder_targets)

		return {self.encoder_inputs: encoder_inputs,
				self.decoder_inputs: decoder_inputs,
				self.decoder_targets: decoder_targets,
				self.encoder_lengths: encoder_lengths,
				self.decoder_lengths: decoder_lengths}

def simple_example():
	vocab = ['a', 'b', '$UNK']

	train = [
		[np.asarray(list('ab')), np.asarray(list('ba'))],
		[np.asarray(list('aab')), np.asarray(list('bba'))],
		[np.asarray(list('abb')), np.asarray(list('baa'))],
		[np.asarray(list('aabb')), np.asarray(list('bbaa'))],
		[np.asarray(list('ba')), np.asarray(list('ab'))],
		[np.asarray(list('baa')), np.asarray(list('abb'))],
		[np.asarray(list('bba')), np.asarray(list('aab'))],
		[np.asarray(list('bbaa')), np.asarray(list('aabb'))]]

	test = [
		[np.asarray(list('ab')), np.asarray(list('ba'))],
		[np.asarray(list('ba')), np.asarray(list('ab'))]]

	seq2seq = TfEncoderDecoder(
		vocab=vocab, max_iter=100, max_length=4)

	X, y = zip(*train)
	seq2seq.fit(X, y)

	X_test, _ = zip(*test)
	print('\nPredictions:', seq2seq.predict(X_test))

if __name__ == '__main__':
	simple_example()
