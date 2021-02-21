
import os
import random

import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        fx = self.__sigmoid(x)
        return fx * (1-fx)
    # Batch generator for mini-batches. Not randomized.

    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=100000, minibatches=True, mbs=100):
        x_batches = self.__batchGenerator(xVals, mbs)
        y_batches = self.__batchGenerator(yVals, mbs)
        for i in range(0, epochs):
            if minibatches is True:
                for x_b, y_b in zip(x_batches, y_batches):
                    L1out, L2out = self.__forward(x_b)
                    L2error = L2out - y_b
                    L2delta = L2error * self.__sigmoidDerivative(L2out)
                    L1error = np.dot(L2delta, self.W2.T)
                    L1delta = L1error * self.__sigmoidDerivative(L1out)
                    self.W1 -= x_b.T.dot(L1delta) * \
                        self.lr * math.exp(-0.1 * i)
                    self.W2 -= L1out.T.dot(L2delta) * \
                        self.lr * math.exp(-0.1 * i)
            else:
                for img in range(0, xVals.shape[0]):
                    L1out, L2out = self.__forward(xVals[[img], :])
                    L2error = L2out - yVals[[img], :]
                    L2delta = L2error * self.__sigmoidDerivative(L2out)
                    L1error = np.dot(L2delta, self.W2.T)
                    L1delta = L1error * self.__sigmoidDerivative(L1out)
                    self.W1 -= xVals[[img], :].T.dot(L1delta) * \
                        self.lr * math.exp(-0.1 * i)
                    self.W2 -= L1out.T.dot(L2delta) * \
                        self.lr * math.exp(-0.1 * i)

    # Forward pass.

    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain = xTrain.reshape((xTrain.shape[0], -1))
    xTest = xTest.reshape((xTest.shape[0], -1))
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        customNet = NeuralNetwork_2Layer(IMAGE_SIZE, 10, 512)
        customNet.train(xTrain, yTrain)
        return customNet
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        # TODO: Write code to build and train your keras neural net.
        print("Not yet implemented.")
        return None
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        return model.predict(data)
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):  # TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    conf_matrix = np.zeros((NUM_CLASSES+2, NUM_CLASSES+2))
    acc = 0
    for i in range(preds.shape[0]):
        pred = np.argmax(preds[i])
        true = np.argmax(yTest[i])
        if pred == true:
            acc = acc + 1
        conf_matrix[pred+1][true+1] += 1
    print(np.sum(conf_matrix, axis=1))
    rowSum = np.sum(conf_matrix, axis=1)
    colSum = np.sum(conf_matrix, axis=0)
    #conf_matrix[NUM_CLASSES+1, 1:] -= [i for i in range(0, )]
    conf_matrix = conf_matrix.tolist()
    conf_matrix[0] = ["TL/PL", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "RowSum"]
    for i in range(1, NUM_CLASSES+2):
        conf_matrix[i][0] = i-1
        conf_matrix[i][NUM_CLASSES+1] = rowSum[i]
        conf_matrix[NUM_CLASSES+1][i] = colSum[i]
    conf_matrix[NUM_CLASSES+1][0] = "ColSum"
    conf_matrix[NUM_CLASSES+1][NUM_CLASSES+1] = np.sum(rowSum)

    F1 = [[], []]
    for i in range(0, NUM_CLASSES):
        pres = conf_matrix[i+1][i+1]/rowSum[i+1]
        rec = conf_matrix[i+1][i+1]/colSum[i+1]
        F1[1].append(round(2/(pres**-1 * rec ** -1), 3))
        F1[0].append(i)

    print('\n'.join(['\t'.join([str(cell) for cell in row])
                     for row in conf_matrix]))
    print('\n'.join(['\t'.join([str(cell) for cell in row])
                     for row in F1]))

    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


# =========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()
