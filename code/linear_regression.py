# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:27:16 2016

@author: katsuya
"""

import theano
import theano.tensor as T
import numpy as np
from sklearn import datasets

class LinearRegression(object):

    def __init__(self, input, n_in):
        self.W = theano.shared(
            value=np.zeros(
                (n_in,),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
            )
        self.b = theano.shared(
            value=np.zeros(
                (),
                dtype=theano.config.floatX
                ),
            name='b',
            borrow=True
            )

        self.y_given_x = T.dot(input, self.W) + self.b

        self.params = [self.W, self.b]

        self.input = input

    def cost(self, y):
        return T.mean((self.y_given_x - y) ** 2)

    def score(self, y):
        u = T.mean((y - self.y_given_x) ** 2)
        v = T.mean(y ** 2) - T.mean(y) ** 2
        return 1 - u / v

boston = datasets.load_boston()

index = np.random.permutation(boston.data.shape[0])

dataset_x = theano.shared(value=boston.data[index,:])
dataset_y = theano.shared(value=boston.target[index])
train_index = index[:int(len(index) * 0.8)]
validate_index = index[int(len(index) * 0.8):int(len(index) * 0.9)]
test_index = index[int(len(index) * 0.9):]

train_set_x = dataset_x[train_index]
train_set_y = dataset_y[train_index]

x = T.matrix('x')
y = T.vector('y')
classifier = LinearRegression(x, 13)
test_model = theano.function(
    inputs=[],
    outputs=classifier.score(y),
    givens=[(x, train_set_x), (y, train_set_y)])

cost = classifier.cost(y)

g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)

learning_rate = 1e-6
updates = [(classifier.W, classifier.W - learning_rate * g_W),
           (classifier.b, classifier.b - learning_rate * g_b)]

train_model = theano.function(
    inputs=[],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x,
        y: train_set_y
        })
for i in range(1000000):
    train_model()
    if i % 10000 == 0:
        print(test_model())