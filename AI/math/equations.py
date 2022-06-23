#!/usr/bin/env python3

import numpy as np

# Y is an array of the labels
# m is the amount of examples that the training data has
# A2 is one of the predictions for what a new example could be

def InitParams():

	# Gives a random weight for each of the 784 inputs
	W1 = np.random.rand(10, 784) - .5
	b1 = np.random.rand(10, 1) - .5
	W2 = np.random.rand(10, 10) - .5
	b2 = np.random.rand(10, 1) - .5

	return W1, b1, W2, b2

def ReLU(Z):
	# Return Z if Z is greater than 0, else return 0
	# Z is a numpy array so this function looks through each element of `Z` and if it is
	# lower than 0, it just returns 0, else it is left untouched
	return np.maximum(Z, 0)

def softmax(Z):
	# Softmax is a function that decides the probability that any particular outcome is the correct outcome
	# This is so that all the numbers are on a spectrum from most likely to least likely, with the most probable result having the largest value on output
	# https://en.wikipedia.org/wiki/Softmax_function
	exp = np.exp(Z - np.max(Z))
	return exp / exp.sum(axis = 0)

def ForwardProp(W1, b1, W2, b2, X):
	# functions that uses forward propagation
	Z1 = W1.dot(X) + b1
	A1 = ReLU(Z1)
	Z2 = W2.dot(A1) + b2
	A2 = softmax(Z2)
	return Z1, A1, Z2, A2

def OneHot(Y): # the Y value being the labels
	# Makes a matrix of zeros with the amount of rows being the size of `Y` (the amount of indexes)
	# With a depth of the maximum amount of array elements
	# Assumes that the classes are in the range [0, 9] (for `Y.max()`) so by adding 1 to that, you get 10 - the ammount of classifications that is wanted
	OneHotY = np.zeros((Y.size, Y.max() + 1))
	# np.arange makes a-range between 0 through `m` (the number of training examples)
	# For each row, go to the column with the label `Y` and set it to `1`
	OneHotY[np.arange(Y.size), Y] = 1
	# Flips is back such that each row is an example (data entry in data set), instead of each column
	OneHotY = OneHotY.T
	return OneHotY

def ReLUPrime(Z):
	return Z > 0 # This returns the slope of `ReLU(Z)` [either 1 or 0 - think about the shape]

def BackProp(Z1, A1, Z2, A2, W2, X, Y):
	# This uses the value of `Y` as learning material and updates the weights (dW*) and biases (db*) to fit accordingly
	# Using Back Propogation (reversing the linear activation function to get new values for weights and biases)
	# It as two different learning variables because of the two layers in the neural network

	# m is the amount of examples that the training data has
	m = Y.size
	OneHotY = OneHot(Y)
	# A2 is one of the predictions
	dZ2 = A2 - OneHotY
	dW2 = 1 / m * dZ2.dot(A1.T)
	db2 = 1 / m * np.sum(dZ2, 1)
	dZ1 = W2.T.dot(dZ2) * ReLUPrime(Z1)
	# Applying the weights in reverse
	# This can be done by using the derivative of the activation function (ReLU)
	dW1 = 1 / m * dZ1.dot(X.T)
	db1 = 1 / m * np.sum(dZ1, 1)
	return dW1, db1, dW2, db2

def UpdateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
	# Updates the parameters with new values, chaning them by the learning rate (`alpha`)
	W1 -= alpha * dW1
	b1 -= alpha * np.reshape(db1, (10, 1))
	W2 -= alpha * dW2
	b2 -= alpha * np.reshape(db2, (10, 1))
	return W1, b1, W2, b2

def GetPredictions(A2):
	return np.argmax(A2, 0)

def GetAccuracy(predictions, Y):
	return np.sum(predictions == Y) / Y.size

def GradientDescent(X, Y, iterations, alpha): # Alpha being the learning rate
	def l():
		print("-" * 36)
	def u(end="\r"):
		print(" Itteration: {} | Accuracy: {:.2f}%".format(i, GetAccuracy(GetPredictions(A2), Y) * 100), end=end)
	l()
	W1, b1, W2, b2 = InitParams()
	for i in range(iterations):
		Z1, A1, Z2, A2 = ForwardProp(W1, b1, W2, b2, X)
		dW1, db1, dW2, db2 = BackProp(Z1, A1, Z2, A2, W2, X, Y)
		W1, b1, W2, b2 = UpdateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
		if i % 5 == 0:
			u()
	u(end="\n")
	l()
	return W1, b2, W2, b2
