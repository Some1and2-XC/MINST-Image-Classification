#!/usr/bin/env python3

# File for Interacting with the Data Sets

import numpy as np
from os import path, chdir, mkdir
from idx2numpy import convert_from_file

from ..wrappers import b

SamplePath = path.abspath("AI/samples") + "\\"

def GetArr(file: str):
	# Takes a Filename and returns as numpy array (Get Array)
	file = SamplePath + file
	return convert_from_file(file)

@b
def MakeCSV(arr1, arr2, name: str = "training-set"):
	# Converts arr1 and arr2 to cvs file where arr1 is the labels and arr2 is the data
	# arr1 is the label set
	# arr2 is the data set
	# To be used after `GetArr()` for both arr variables
	n = arr2.shape
	if len(n) != 3:
		raise TypeError("Incorrect Shape of Arrray")

	else:
		text = "\n".join(f"{arr1[i]}," + ",".join(",".join(str(arr2[i][j][k]) for k in range(n[2])) for j in range(n[1])) for i in range(n[0]))

		directory = SamplePath
		try:
			chdir(directory)
		except FileNotFoundError:
			mkdir(directory)
			chdir(directory)

		with open(f"{name}.csv", "w") as f:
			f.write(text)
