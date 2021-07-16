#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mysklearn.linearmodels import LogisticRegression
from model_selection import train_test_split

# load the data
path = "../data/LogR/"
X = np.load(path+"X.npy")
y = np.load(path+"y.npy")
print(X.shape, y.shape)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# train test split
X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# create instance/ object of my model
logr = LogisticRegression()

# training
losses = logr.fit(X_train, y_train, learning_rate=0.001, epochs=50)
plt.plot(losses)
plt.show()

logr.w

# evaluate your model on test/ validation set
ypred = logr.predict(X_test) 
test_acc = logr.accuracy(y_test, ypred)
print(test_acc)


# ## Plot the decision Boundary
# z = m1.x1 + m2.x2 + c  
# On the feature plane/ space, z=0  
# x2 = (-m1.x1 - c )/m2


# generate some points for x1 feature
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 20)
m1, m2, c = logr.w[1][0], logr.w[2][0], logr.w[0][0]
x2 = (-m1*x1 - c )/m2

plt.plot(x1, x2) # decision boundary
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=plt.cm.Set1) # plot
plt.show()


# # Visualize the Z surface

def predict_Z(w, X):
    """ 
    w : column vector: (n+1,1)
    X: feature vector of 1 pt (m, n)
    """
    return w[0][0] + w[1][0]*X[0] + w[2][0]*X[1]

def sigmoid(z):
    return 1/(1+ (np.e**(-z)) ) 

X1 = np.linspace(X[:,0].min(), X[:,0].max(), 50)
X2 = np.linspace(X[:,1].min(), X[:,1].max(), 50)
X1, X2 = np.meshgrid(X1, X2)
print(X1.shape, X2.shape)

z = []
for tp in zip(X1.reshape(-1,1), X2.reshape(-1,1)):
    z.append(predict_Z(logr.w, tp))
z = np.array(z).reshape(X1.shape)


plt.figure(figsize=(7,7))
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, z, cmap=plt.cm.coolwarm, alpha=0.5)
ax.scatter(X[:,0], X[:,1], np.zeros(X[:,0].shape), c=y) # data points

ax.plot_surface(X1, X2, sigmoid(z), cmap=plt.cm.rainbow)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Z")

ax.set_zlim(-2,2)

plt.show()