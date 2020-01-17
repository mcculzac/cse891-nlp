"""
Zachary McCullough
mccul157@msu.edu
2020-01-16
Class to contain perceptron class
"""

#########
# imports
#########

import numpy as np


#########
# classes
#########

class Perceptron:

    def __init__(self, size=10):
        self.weights = np.zeros(size)

    def predict(self, inp: np.array):
        dotted = np.multiply(inp, self.weights)
        predictions = np.zeros(self.weights.shape)
        for k, d in enumerate(dotted):
            if d > 20:
                predictions[k] = 20
            if d < 1:
                predictions[k] = 1
            else:
                predictions[k] = int(d)
        classes = {}
        for val in predictions:
            if val not in classes:
                classes[val] = 1
            else:
                classes[val] += 1
        maxy = -1
        key = -1
        for k in classes:
            if classes[k] > maxy:
                key = k
                maxy = classes[k]
        y_prime = key

        return y_prime, predictions

    def train(self, data: np.array, labels: np.array, lr: float, epochs: int=1000):
        N = len(set(labels))
        for _ in range(epochs):
            print('Epoch', _)
            convergence = True
            for k, point in enumerate(data):
                y_prime, vec = self.predict(point)
                label = labels[k]
                for k, i in enumerate(vec):
                    if vec[k] != label:
                        self.weights[k] += lr * (label - y_prime)




