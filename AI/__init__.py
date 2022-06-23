#!/usr/bin/env python3

from os import path

import numpy as np
import pandas as pd

from .services.data import GetArr
from .services.data import MakeCSV
from .math.equations import GradientDescent
from .predictions.predictions import TestPrediction
from .saves.save import SaveData, LoadData
# from .turtles.turtles import board

def MakeCSVFromRaw():
	# Makes `.csv` file out of raw data if True
	MakeCSV(GetArr("DataSet/t10k-labels.idx1-ubyte"), GetArr("DataSet/t10k-images.idx3-ubyte"), name = "testing-set")
	MakeCSV(GetArr("DataSet/train-labels.idx1-ubyte"), GetArr("DataSet/train-images.idx3-ubyte"), name = "training-set")

def WriteInformation(txt: str, end="\n"):
	# Writes Updates
	print(f" -- {txt}", end = end)

class ai:
	def __init__(self, iterations: int = 1000, amnt_of_examples: int = 1000):
		self.iterations = iterations
		self.amnt_of_examples = amnt_of_examples
		self.path_to_sample_folder = "AI/samples/"
		self.training_file_name = path.abspath(f"{self.path_to_sample_folder}training-set.csv")
		self.testing_file_name = path.abspath(f"{self.path_to_sample_folder}testing-set.csv")

	def train(self):

		self.InitialiseData()

		WriteInformation("Array Dimensions: [ {} x {} ]".format(*self.data_train.shape))

		def cp(arr):
			return np.copy(arr) #, dtype=np.float128)

		self.W1, self.b1, self.W2, self.b2 = GradientDescent(cp(self.X_train), cp(self.Y_train), self.iterations, 0.1)

	def InitialiseData(self):
		# Imports data from both csv files

		WriteInformation("Importing Data")
		WriteInformation("[0 / 2] - Importing", end="\r")
		self.data_train = pd.read_csv(self.training_file_name)
		self.data_train.head()
		self.data_train = np.array(self.data_train)
		WriteInformation("[1 / 2] - Importing", end="\r")
		self.data_dev = pd.read_csv(self.testing_file_name)
		self.data_dev.head()
		self.data_dev = np.array(self.data_dev)
		WriteInformation("[2 / 2] - Importing")

		# Gets the dimensions of the data
		_, n = self.data_train.shape

		# Randomizes the order of the data
		np.random.shuffle(self.data_train)

		self.SeparateData(n) # Where the n that is passed is the number of columns (each data index)

	def SeparateData(self, n: int):
		# Sets variables to sections of the data

		# `.T` Transposes List - basically switches x and y axis
		self.data_train = self.data_train[:self.amnt_of_examples].T
		self.Y_train = self.data_train[0]
		self.X_train = self.data_train[1:]
		self.X_train = self.X_train / 255.

		self.data_dev = self.data_dev.T
		self.Y_dev = self.data_dev[0]
		self.X_dev = self.data_dev[1:n]
		self.X_dev = self.X_dev / 255.

		_, self.m_train = self.X_train.shape

	def save(self):
		# Function for saving model created
		n = SaveData(self.W1, self.b1, self.W2, self.b2)
		WriteInformation(f"Model `{n}` Saved")

	def load(self, name: str = None):
		if type(name) != str:
			raise TypeError("Name must be a string!")
		self.InitialiseData()
		self.W1, self.b1, self.W2, self.b2 = LoadData(n = name)

	def test(self, n: int = 1):
		1
		TestPrediction(self.X_dev, self.Y_dev, n, self.W1, self.b1, self.W2, self.b2)
