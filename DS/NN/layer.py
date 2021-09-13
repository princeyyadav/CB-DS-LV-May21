import numpy as np

class Dot:

    def __init__(self, input_size, units):
        self.W = np.random.randn(input_size, units)
        self.b = np.random.randn(1, units)

    def __call__(self, X):
        return self.eval(X)

    def eval(self, X):
        return X.dot(self.W) + self.b

    def grad_input(self, X):
        return np.stack([self.W.T]*X.shape[0], axis=0) 

    def grad_w(self, X):
        I = np.identity(self.b.shape[1])
        m1 = np.stack([I]*self.W.shape[0], axis=1)
        return np.einsum('ijk,mj->mijk', m1, X)

    def grad_b(self, X):
        return np.stack([np.identity(self.b.shape[1])]*X.shape[0], axis=0)
    
    def get_parameter_shape(self):
        return self.W.shape, self.b.shape


class Dense:

    def __init__(self, input_size, activation, units):
        """
        input_size: no. of neurons in previous layer
        activation: some activation funtion
        units: no. of neurons in current layer 
        """
        self.activation = activation
        self.units = units
        self.dot = Dot(input_size, units)

    def eval(self, X):
        return self.activation(self.dot(X))

    def grad_input(self, X):
        g1 = self.activation.grad_input( self.dot(X) )
        g2 = self.dot.grad_input(X)
        return np.einsum('mij,mjk->mik', g1, g2)

    def grad_parameters(self, X):
        da_dI = self.activation.grad_input(self.dot(X))
        dI_dw = self.dot.grad_w(X)
        da_dw = np.einsum('mij,mjkl->mikl', da_dI, dI_dw)

        dI_db = self.dot.grad_b(X)
        # print(da_dI.shape, dI_dw.shape, dI_db.shape)
        da_db = np.einsum('mij,mjk->mik',  da_dI, dI_db)
        return da_dw, da_db

    def backprop_grad(self, grad_loss, grad):
        dL_dwi = np.einsum('mij,mjkl->mikl', grad_loss, grad['w']).sum(axis=0)
        dL_dbi = np.einsum('mij,mjk->mik', grad_loss, grad['b']).sum(axis=0)
        grad_loss = np.einsum('mij,mjk->mik', grad_loss, grad['input'])
        return dL_dwi, dL_dbi, grad_loss
        
    def update(self, grad, optimizer):
        """ grad: (dL_dwi, dL_dbi)"""
        self.dot.W = optimizer.minimize(self.dot.W, grad[0])
        self.dot.b = optimizer.minimize(self.dot.b, grad[1])
        
    def get_parameter_shape(self):
        return self.dot.get_parameter_shape()
    
    def get_total_parameters(self):
        w_shape, b_shape = self.dot.get_parameter_shape()
        return np.prod(w_shape) + np.prod(b_shape)