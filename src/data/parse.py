"""
Parsing code based on https://github.com/futurulus/negotiation/.
Data from FAIR "Deal or no deal?" dataset is stored in data/raw/.

See also:
https://github.com/facebookresearch/end-to-end-negotiator/blob/master/src/data.py
"""

input_directory = "data/raw/"
output_directory = "data/processed/"
files = ["train", "val", "test"]

"""
Special tokens:
 - <eos>: end of speech
 - <unk>: unknown word
 - <selection>: selection
 - <pad>: padding
""" 
SPECIAL_TOKENS = ["<eos>", "<unk>", "<selection>", "<pad>"]
STOP_TOKENS = ["<eos>", "<selection>"]


def read_lines(filename):
	lines = []
	with open(filename, 'r') as f:
		for line in f:
			lines.append(line.strip())
	return lines


def main():
	for file in files:
		input_filename = input_directory + file + ".txt"
		output_filename = output_directory + file + ".txt"

		lines = read_lines(input_filename)
		for line in lines:
			tokens = line.split(" ")
			
	