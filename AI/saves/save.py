#!/usr/bin/env python3

import numpy as np
from os import chdir, mkdir, path
from random import randint

from ..wrappers import b

def cd():
	p = path.abspath("AI/models")
	try:
		chdir(p)
	except FileNotFoundError:
		mkdir(p)
		chdir(p)

@b
def SaveData(W1, b1, W2, b2):
	# Function for saving a model
	cd()
	n = "".join(chr(randint(65, 90)) for i in range(4)) # Generates a prefix to each file saved
	np.save(f"{n}-W1.npy", W1)
	np.save(f"{n}-b1.npy", b1)
	np.save(f"{n}-W2.npy", W2)
	np.save(f"{n}-b2.npy", b2)
	return n

@b
def LoadData(n):
	# Function for loading a saved model

	# If this breaks because something doesn't exist the
	# exception will fly all the way to the top so I don't have to worry about it
	cd()
	W1 = np.load(f"{n}-W1.npy")
	b1 = np.load(f"{n}-b1.npy")
	W2 = np.load(f"{n}-W2.npy")
	b2 = np.load(f"{n}-b2.npy")
	return W1, b1, W2, b2