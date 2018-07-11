import numpy as np
import tensorflow as tf
from tf_rnn_classifier import TfRNNClassifier
import warnings
import random
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense


__author__ = 'Nicholas Tomlin'

class TfEncoderDecoder(TfRNNClassifier):
	"""
	Parameters
	----------
	max_input_length : int
		Maximum sequence length for the input.
	max_output_length : int
		Maximum sequence length for the output.
	num_layers : int
		Number of layers in the RNN. Used for encoder and decoder. 
	vocab : list
		The full vocabulary. `prepare_data()` will convert the data provided
		to `fit` and `predict` methods into a list of indices into this
		list of items. For now, assume the input and output have the
		same vocabulary.
	"""
	def __init__(self, max_input_length=7, max_output_length=8, num_layers=2, **kwargs):
		self.max_input_length = max_input_length
		self.max_output_length = max_output_length
		self.num_layers = num_layers

		super(TfEncoderDecoder, self).__init__(**kwargs)


	def build_graph(self):
		"""
		Builds a single graph for training and inference.
		"""
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
		"""
		Builds the embedding space, and returns embeddings for both the 
		encoder and the decoder inputs.
		"""
		self.embedded_encoder_inputs = tf.contrib.layers.embed_sequence(
			ids=self.encoder_inputs,
			vocab_size=self.vocab_size,
			embed_dim=self.embed_dim)

		self.decoder_embedding_space = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim]))
		self.embedded_decoder_inputs = tf.nn.embedding_lookup(
			self.decoder_embedding_space,
			self.decoder_inputs)

	def encoding_layer(self):
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


	def decoding_layer(self):
		"""
		Two separate decoders for training and inference (prediction): inference
		reuses weights from training during predict().
		"""
		self.decoding_training()
		self.decoding_inference()


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


	def decoding_inference(self):
		start_tokens = tf.tile(
			input=tf.constant([self.vocab.index("<START>")], dtype=tf.int32),
			multiples=[self.batch_size])

		inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
			embedding=self.decoder_embedding_space,
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


	def prepare_output_data(self, y):
		"""
		Modified to treat y as a sequence. Avoids one-hot
		encoding of y in fit(). Use prepare_data() instead.
		"""
		return y


	def prepare_data(self, X, max_length):
		"""
		Modification of _convert_X that takes max_length as a parameter.
		This is useful because inputs and outputs may have different
		max_lengths in the encoder-decoder model.
		"""
		new_X = np.zeros((len(X), max_length), dtype='int')
		ex_lengths = []
		index = dict(zip(self.vocab, range(len(self.vocab))))
		unk_index = index['$UNK']
		for i in range(new_X.shape[0]):
			ex_len = min([len(X[i]), max_length])
			ex_lengths.append(ex_len)
			vals = X[i][-max_length: ]
			vals = [index.get(w, unk_index) for w in vals]
			temp = np.zeros((max_length,), dtype='int')
			temp[0: len(vals)] = vals
			new_X[i] = temp
		return new_X, ex_lengths


	def get_cost_function(self, **kwargs):
		# With built-in Tensorflow seq2seq loss:
		masks = tf.sequence_mask(self.decoder_lengths, self.max_output_length, dtype=tf.float32, name='masks')
		cost = tf.contrib.seq2seq.sequence_loss(
			logits=self.training_logits,
			targets=self.decoder_targets,
			weights=masks)
		return cost


	def predict(self, X):
		X, x_lengths = self.prepare_data(X, self.max_input_length)
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


	def output(self, answer_logits):
		"""
		TODO: Convert answer_logits to printed output by indexing into vocabulary.
		"""
		pass


	def train_dict(self, X, y):
		decoder_inputs = [["<START>"] + list(seq) for seq in y]
		decoder_targets = [list(seq) + ["<END>"] for seq in y]

		encoder_inputs, encoder_lengths = self.prepare_data(X, self.max_input_length)
		decoder_inputs, _ = self.prepare_data(decoder_inputs, self.max_output_length)
		decoder_targets, decoder_lengths = self.prepare_data(decoder_targets, self.max_output_length)

		return {self.encoder_inputs: encoder_inputs,
				self.decoder_inputs: decoder_inputs,
				self.decoder_targets: decoder_targets,
				self.encoder_lengths: encoder_lengths,
				self.decoder_lengths: decoder_lengths}

	def get_optimizer(self):
		# Adagrad optimizer:
		return tf.train.AdagradOptimizer(self.eta).minimize(self.cost)


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
	print('\nPredictions:', seq2seq.predict(X_test))

if __name__ == '__main__':
	simple_example()
