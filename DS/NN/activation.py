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