# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-03-07 03:40:40
# @Last Modified by:   shubham
# @Last Modified time: 2017-03-07 04:13:11

import numpy as np

class RNN():
	def __init__(self, insize, hsize, outsize, learning_rate=0.1):

		# [h x 1] hidden state, summarising the contents of past
		self.h = np.zeros((hsize, 1))

		# weights, normally distributed
		self.Wxh = np.random.randn((insize, hsize)) * 0.01
		self.Whh = np.random.randn((hsize, hsize)) * 0.01
		self.Why = np.random.randn((hsize, outsize)) * 0.01

		# biases
		self.bh = np.zeros((hsize, 1))
		self.by = np.zeros((outsize, 1))

		
