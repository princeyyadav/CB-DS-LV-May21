import numpy as np

class Sigmoid:

    def __call__(self, X):
        return self.eval(X)

    def eval(self, X):
        return 1/((np.e**-X) + 1)

    def grad_input(self, X):
        I = np.identity(X.shape[1])
        b = self.eval(X)*(1-self.eval(X)) # same shape as X
        return np.einsum('ij,mi->mij', I, b)

class ReLU:

    def __call__(self, X):
        return self.eval(X)

    def eval(self, X):
        return X*(X>=0)

    def grad_input(self, X):
        I = np.identity(X.shape[1])
        b = np.ones(X.shape) # same shape as X
        b[X<0] = 0
        return np.einsum('ij,mi->mij', I, b)