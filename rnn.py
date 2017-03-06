# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-03-07 03:40:40
# @Last Modified by:   shubham
# @Last Modified time: 2017-03-07 05:08:28

import sys
import numpy as np

class RNN():
	def __init__(self, insize, hsize, outsize, learning_rate=0.1):

		# voacb size
		self.vocab_size = insize

		# [h x 1] hidden state, summarising the contents of past
		self.h = np.zeros((hsize, 1))

		# weights, normally distributed
		self.Wxh = np.random.randn(insize, hsize) * 0.01
		self.Whh = np.random.randn(hsize, hsize) * 0.01
		self.Why = np.random.randn(hsize, outsize) * 0.01

		# biases
		self.bh = np.zeros((hsize, 1))
		self.by = np.zeros((outsize, 1))

		self.lr = learning_rate

	def step(self, inputs, targets):
		x, h, y, p = {}, {}, {}, {}

		loss = 0
		h[-1] = np.copy(self.h)

		# for each time-step
		for t in range(len(x)):
			# one-hot-encode input
			x[t] = np.zeros((1, vocab_size))
			x[t][inputs[t]] = 1

			# tanh(x*w1 + h_prev*w1 + b) -- hidden state
			h[t] = np.tanh(np.dot(x, self.Wxh) + np.dot(self.h[t-1]) + self.bh)

			# unnormalized log probabilities for next char
			y[t] = np.dot(h[t], self.Why) + self.by

			# probability distribution, softmax loss
			p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))
			loss += -p[t][targets[t]]

		return loss


def main():
	# should be simple plain text file
	data = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read()
	chars = list(set(data))

	data_size, vocab_size = len(data), len(chars)
	print('File has {} characters with {} unique.'.format(data_size, vocab_size))

	# make some dictionaries for encoding and decoding from 1-of-k
	char_to_ix = { ch:i for i,ch in enumerate(chars) }
	ix_to_char = { i:ch for i,ch in enumerate(chars) }

	# network arch for RNN
	insize, outsize = vocab_size, vocab_size
	hsize = 100
	
	rnn = RNN(insize, hsize, outsize)

	# number of steps to unroll the RNN for
	seq_length = 25 
	
	i, n = 0, 0		
	while True:
		inputs = [char_to_ix[ch] for ch in data[i:i+seq_length]]
		targets = [char_to_ix[ch] for ch in data[i+1:i+seq_length+1]]


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print('\nKeyboardInterrupt')

