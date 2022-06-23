#!/usr/bin/env python3

from matplotlib import pyplot as plt

from random import randint

from ..math.equations import ForwardProp, GetPredictions

def MakePredictions(X, W1, b1, W2, b2):
    # Function for getting a predictions from an image
    _, _, _, A2 = ForwardProp(W1, b1, W2, b2, X)
    predictions = GetPredictions(A2)
    return predictions

def TestPrediction(X, Y, n, W1, b1, W2, b2):
    m = Y.size
    for i in range(n):
        index = randint(0, m)
        prediction = MakePredictions(X[:, index, None], W1, b1, W2, b2)
        label = Y[index]
        current_image = X[:, index, None]
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.title(f"Prediction: {prediction}")
        plt.xlabel(f"Label: {label}")
        plt.imshow(current_image, interpolation='nearest')
        plt.show()