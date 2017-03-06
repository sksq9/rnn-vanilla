# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-03-07 03:40:40
# @Last Modified by:   shubham
# @Last Modified time: 2017-03-07 04:50:17

import numpy as np

class RNN():
	def __init__(self, insize, hsize, outsize, learning_rate=0.1):

		# voacb size
		self.vocab_size = insize

		# [h x 1] hidden state, summarising the contents of past
		self.h = np.zeros((hsize, 1))

		# weights, normally distributed
		self.Wxh = np.random.randn((insize, hsize)) * 0.01
		self.Whh = np.random.randn((hsize, hsize)) * 0.01
		self.Why = np.random.randn((hsize, outsize)) * 0.01

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

