import numpy as np

class BinaryCrossEntropy:

    def __call__(self, ytrue, ypred):
        """ 
        ytrue: column vector (m,1)
        ypred: column vector (m,1)
        Returns: Loss (1,1)
        """
        assert ytrue.shape == ypred.shape, f"shape mismatch ytrue {ytrue.shape} != ypred {ypred.shape}"
        return -np.sum( ytrue*np.log(ypred + 1e-10) + (1-ytrue)*np.log(1-ypred + 1e-10) )

    def grad_input(self, y, ypred):
        grad = np.zeros(ypred.shape)
        ix0 = (y==0).reshape(-1,)
        ix1 = (y==1).reshape(-1,)
        grad[ix0,:] = 1/(1-ypred[ix0,:])
        grad[ix1,:] = -1/ypred[ix1,:]
        return grad.reshape(-1,1,1)