"""
Parsing code based on https://github.com/futurulus/negotiation/.
Data from FAIR "Deal or no deal?" dataset is stored in data/raw/.

See also:
https://github.com/facebookresearch/end-to-end-negotiator/blob/master/src/data.py
"""
from collections import Counter

"""
Special tokens:
 - <eos>: end of speech
 - <unk>: unknown word
 - <selection>: selection
 - <pad>: padding
"""

SPECIAL_TOKENS = ["<eos>", "<unk>", "<selection>", "<pad>"]
STOP_TOKENS = ["<eos>", "<selection>"]


class Parser(object):
	"""
	Responsible for parsing the raw FB negotiation data. Separates each
	dialogue instance into two examples (from each perspective).

	TODO: further separate into smaller examples for response model (?)

	Parameters
	----------
	input_directory : string
		Location of unprocessed FB negotiation data.
	output_directory : string
		Location of parsed training examples.
	unk_threshold : int
		Tokens appearing less than unk_threshold times will be UNKed.
		Set to 20 in original FB experiments.
	"""
	def __init__(self,
		input_directory="../../data/raw/",
		output_directory="../../data/processed/",
		unk_threshold=20):
		self.input_directory = input_directory
		self.output_directory = output_directory
		self.unk_threshold = unk_threshold

		get_counts()
		get_vocab()

	def get_counts(self):
		"""
		Get the frequency counts for each word in data.txt.
		Used to create a vocabulary list and determine UNKing.
		"""
		self.counts = Counter()
		with open(input_directory + "data.txt") as f:
			for line in f:
				line = line.strip()
				tokens = line.split(" ")
				for token in tokens:
					self.counts[token] += 1


	def get_vocab(self):
		"""
		Returns a list of every character which occurs more than
		self.unk_threshold times

		With unk_threshold of 20, vocab_size is 544.
		"""
		self.vocab = ["<PAD>", "$UNK", "<START>", "<END>"]

		for token in self.counts:
			num_occurrences = self.counts[token]
			if num_occurrences >= self.unk_threshold:
				self.vocab.append(token)

		self.vocab_size = len(self.vocab)


	def parse(self):
		"""
		Iterates through raw data files and calls relevant helpers.
		"""
		split = ["train", "val", "test"]
		for file in split:
			input_filename = input_directory + file + ".txt"
			output_filename = output_directory + file + ".txt"
			parse_file(input_filename, output_filename)


	def parse_file(self, input_filename, output_filename):
		"""
		Converts each line of input_filename into training examples, which
		are written to output_filename.
		"""
		with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
			for line in input_file:
				training_examples = self.create_training_examples(line)
				for example in training_examples:
					output_file.write(example)


	def create_training_examples(self, line):
		"""
		Creates a single training example from a line of raw data.
		"""
		print(line)


def read_lines(filename):
	lines = []
	with open(filename, 'r') as f:
		for line in f:
			lines.append(line.strip())
	return lines

def main():

	
		print(input_filename)
		lines = read_lines(input_filename)
		for line in lines:
			print(line)
			tokens = line.split(" ")
			

if __name__ == '__main__':
	parser = Parser()
	parser.parse()