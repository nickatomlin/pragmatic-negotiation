class Agent(TfEncoderDecoder):
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