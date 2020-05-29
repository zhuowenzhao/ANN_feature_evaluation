#!/usr/bin/env python3

# author: Zhuowen Zhao
# LicenseCC BY-SA 4.0
# Please cite: https://doi.org/10.1016/j.scriptamat.2020.04.029

import numpy as np
import os,sys,string,xlrd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler 
    scaler = StandardScaler()  
    scaler.fit(X_train) 
    X_train_norm = scaler.transform(X_train)  
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

def vectorize_y(label):
    la = np.empty([1,2])
    for i in range(len(label)):
      if label[i] == -1:
        la = np.vstack((la,np.array([1,0]).reshape(1,2)))
      elif label[i] == 1:
        la = np.vstack((la,np.array([0,1]).reshape(1,2)))
    return la[1:,:].astype(int)
      
def softmax(z):       
    exp_value = np.exp(z-np.amax(z, axis=1, keepdims=True)) # for stablility
    softmax_scores = exp_value / np.sum(exp_value, axis=1, keepdims=True)
    return softmax_scores

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

def relu(x):
    return np.maximum(0,x)

def dRelu(x):
    ones = np.ones(x.shape)
    x_new = np.where(x<0,x,ones)
    return np.maximum(0,x_new)

class twolayers_NN:
    '''
    Gradient descent used for backpropagation update 
    '''
    def __init__(self, X, y, hidden_layer_nn=50, lr=0.0001): 
        self.X = X # features
        self.y = y 
        self.hidden_layer_nn = hidden_layer_nn 
        self.lr = lr 
        
        # Initialize weights 
        self.nn = X.shape[1] 
        self.W1 = np.random.randn(self.nn, hidden_layer_nn) / np.sqrt(self.nn)  
        self.b1 = np.zeros((1, hidden_layer_nn)) 
        self.output_layer_nn = y.shape[1]
        self.W2 = np.random.randn(hidden_layer_nn, hidden_layer_nn) / np.sqrt(hidden_layer_nn)
        self.b2 = np.zeros((1, hidden_layer_nn))
        self.W3 = np.random.randn(hidden_layer_nn, self.output_layer_nn) / np.sqrt(hidden_layer_nn)
        self.b3 = np.zeros((1, self.output_layer_nn))

    def feed_forward(self):

        self.z1 = self.X @ self.W1 + self.b1
        self.f1 = relu(self.z1)
        self.df1dz1 = dRelu(self.z1)
        self.z2 = self.f1 @ self.W2 + self.b2
        self.f2 = relu(self.z2)
        self.df2dz2 = dRelu(self.z2)
        self.z3 = self.f2 @ self.W3 + self.b3
        self.y_hat = softmax(self.z3)
        
    def back_propagation(self):
        d3 = self.y_hat - self.y
        dW3 = self.f2.T @ d3                                                              
        db3 = np.sum(d3, axis=0, keepdims=True)                                           
        d2 = (d3 @ self.W3.T)*self.df2dz2
        dW2 = self.f1.T @ d2
        db2 = np.sum(d2, axis=0, keepdims=True)
        d1 = (d3 @ self.W3.T @ self.W2.T)*self.df2dz2*self.df1dz1
        dW1 = self.X.T @ d1
        db1 = np.sum(d1, axis=0, keepdims=True)
        
        # Update the gradident descent
        self.changeW = np.linalg.norm(dW1,1)+np.linalg.norm(dW2,1)+np.linalg.norm(dW3,1)
        self.changeb = np.linalg.norm(db1,1)+np.linalg.norm(db2,1)+np.linalg.norm(db3,1)
        self.W1 -= self.lr * dW1
        self.b1 = self.b1 - self.lr * db1
        self.W2 = self.W2 - self.lr * dW2
        self.b2 = self.b2 - self.lr * db2
        self.W3 = self.W3 - self.lr * dW3
        self.b3 = self.b3 - self.lr * db3
       
        
    def cross_entropy_loss(self):
        self.feed_forward()
        self.loss = -np.sum(self.y*np.log(self.y_hat + 1e-6))   # L2 norm
        
    def predict(self, X_test):
        z1 = X_test @ self.W1 + self.b1
        f1 = relu(z1)
        z2 = f1 @ self.W2 + self.b2
        f2 = relu(z2)
        z3 = f2 @ self.W3 + self.b3    
        y_hat_test = softmax(z3)
        labels = [-1, 1]
        num_test_samples = X_test.shape[0]
        ypred = np.zeros(num_test_samples, dtype=int) 
        for i in range(num_test_samples):
            ypred[i] = labels[np.argmax(y_hat_test[i,:])] 
        return ypred

# main
fname = sys.argv[-1]
data = np.array(pd.read_excel(fname, sheet_name='data'))

features = [1,2,3,4,5,6,7,8,9]  # feature indices

# mask = np.any(np.isnan(fdata), axis=1)
# data=fdata[~mask]

# exclude labels 0.5
mask_non = data[:,10] == 0
data = data[~mask_non]

combos = np.array([[1],
                   [2],
                   [3],      
                   [4],      
                   [5],
                   [6],
                   [7],
                   [8],
                   [9] ])

with open('outputs.log','w') as file:
  file.write('\n ++++++++++++++++++++++++++++++++++++++++++++++++\n')
  file.close()


with open('outputs.log','a') as file:
  ks = [0.3]
  for k in ks:
    print('k {}'.format(k))
    file.write('\n ++++++++++++++++++Test size {}+++++++++++++++++++\n'.format(k))
    for l in range(len(combos)):
      print(combos[l])
      file.write('\n *****features selection {}*****'.format(combos[l]))
      acc = []
      for j in range(20):
        print('doing round {}...'.format(j+1))
        X_train, X_test, y_train, y_test = train_test_split(data[:,:10], data[:,10], test_size = k, random_state = None)
        X_train_norm,X_test_norm = normalize_features(X_train,X_test)
        y_train_ohe = vectorize_y(y_train)
        y_test_ohe  = vectorize_y(y_test)
        X_train_select = X_train_norm[:,combos[l]]
        X_test_select  = X_test_norm[:,combos[l]]
        myNN = twolayers_NN(X_train_select, y_train_ohe, hidden_layer_nn=20, lr=0.0001)   
        epoch_num = 2000
        for i in range(epoch_num):
            myNN.feed_forward()
            myNN.back_propagation()
            myNN.cross_entropy_loss()        
      
        y_pred = myNN.predict(X_test_select)
        acc.append(accuracy(y_pred, y_test.ravel()))
      print('average accuracy {}, std {}'.format(np.mean(acc),np.std(acc)))
      file.write('\n Average accuracy {}, std {}\n'.format(np.mean(acc),np.std(acc)))