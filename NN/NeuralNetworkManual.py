import numpy as np
import formulas 
import pandas as pd
df = pd.DataFrame()
from sklearn.utils import shuffle

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
    return s * (1 - s)

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def MSE(pred, real):
    return ((pred-real)**2).mean(axis=1)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

class NeuralNetwork:
    def __init__(self, formula, neurons, lr, hidden_layers, activation, error):
        self.data = formula.data
        self.labels = formula.labels

        self.data, self.labels = shuffle(self.data, self.labels, random_state = 0)
        self.train, self.validate, self.test = np.split(self.data, [int(.6*len(df)), int(.8*len(df))])
        self.train_labels, self.validate_labels, self.test_labels = np.split(self.labels, [int(.6*len(df)), int(.8*len(df))])

        self.neurons = neurons 
        self.lr = lr
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.error = error


        
        input_dim = self.data.shape[1]
        output_dim = self.labels.shape[1]
        

        self.w1 = np.random.randn(input_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, output_dim)
        self.b3 = np.zeros((1, output_dim))
    
    def feedforward(self):
        z1 = np.dot(self.train, self.w1) + self.b1
        self.a1 = self.activation(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.activation(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.activation(z3)
    
    def backprop(self):
        loss = error(self.a3, self.labels)
        print('Error :', loss)
        a3_delta = cross_entropy(self.a3, self.labels) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.train.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

model = NN() 

def get_acc(data, labels):
    acc = 0
    for x,y in zip(data, labels):
        s = model.predict(x)
        if s == np.argmax(y):
            acc +=1
    return acc/len(data)*100

def summary(self, train, val):
    print("Training accuracy : ", get_acc(self.train/16, np.array(self.train_labels)))
    print("Test accuracy : ", get_acc(self.val/16, np.array(self.val_labels)))

def predict(self, data):
    self.x = data
    self.feedforward()
    return self.a3.argmax()
