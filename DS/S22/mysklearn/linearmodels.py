import numpy as np

class LogisticRegression:

    def __get_gradient(self, X, y):
        gradient = []
        for  i in range(X.shape[1]):
            grad = -2*( ( (y - self.predict(X))* (X[:,i].reshape(-1,1)) ).sum() ) 
            gradient.append(grad)
        return np.array(gradient).reshape(-1,1)

    def __gradient_descent(self, X, y, learning_rate, epochs, batch_size):

        # start with any random weights
        self.w = np.random.randn(X.shape[1]).reshape(-1,1)
        # self.w = np.zeros((X.shape[1], 1)) 

        idx = np.arange(0,X.shape[0])

        losses = []

        for i in range(epochs):
            random_idx = np.random.choice(idx, size = batch_size)

            # update rule
            self.w = self.w - learning_rate*self.__get_gradient(X[random_idx], y[random_idx])

            ypred = self.predict(X)
            loss = self.loss(y, ypred)
            acc = self.accuracy(y, ypred)
            losses.append(loss)
            print("\r" + f"epoch: {i}, loss: {loss}, accuracy: {acc}", end="") 

        return losses

    def fit(self, X, y, method="batch", learning_rate=0.001, epochs=300, **kwargs):
        """ Training the model"""
        X = X.copy()
        ones_column = np.ones((len(X),1))
        X = np.concatenate([ones_column, X], axis=1)

        if method == "batch":
            batch_size = X.shape[0] # all the samples

        elif method == "mini-batch":
            if kwargs.get('batch_size') == None:
                batch_size = int(X.shape[0]*0.25)
            else:
                batch_size = kwargs['batch_size']

        elif method == 'stochastic':

            batch_size = 1

        return self.__gradient_descent(X, y, learning_rate, epochs, batch_size)
        
        
    def sigmoid(self, z):
        return 1/ (1 + np.e**(-z))

    def predict(self, X):
        """ X: Feature matrix"""
        if X.shape[1] != self.w.shape[0]:
            X = X.copy()
            ones_column = np.ones((len(X),1))
            X = np.concatenate([ones_column, X], axis=1)

        return self.sigmoid(X.dot(self.w))

    def loss(self, y, ypred):
        # print("y ypred shape", y.shape, ypred.shape)
        # get indices with y = 1
        ones_idx = y==1
        zero_idx = y==0

        ones_loss = (y[ones_idx]*np.log(ypred[ones_idx] + 1e-10) ).sum(axis=0)
        zeros_loss = ( (1-y[zero_idx]) * np.log(1-ypred[zero_idx] + 1e-10) ).sum(axis=0)
        # print("loss shape", zeros_loss.shape, ones_loss.shape)

        return -1 * (ones_loss + zeros_loss)

    def accuracy(self, ytrue, ypred):
        return  ((ypred > 0.5).astype('int') == ytrue).mean()



class LinearRegression:

    def __get_gradient(self, X, y):
        gradient = []
        for  i in range(X.shape[1]):
            grad = -2*( ( (y - self.predict(X))* (X[:,i].reshape(-1,1)) ).sum() ) 
            gradient.append(grad)
        return np.array(gradient).reshape(-1,1)

    def __gradient_descent(self, X, y, learning_rate, epochs, batch_size):

        # start with any random weights
        self.w = np.random.randn(X.shape[1]).reshape(-1,1)
        # self.w = np.zeros((X.shape[1], 1)) 

        idx = np.arange(0,X.shape[0])

        losses = []

        for i in range(epochs):
            random_idx = np.random.choice(idx, size = batch_size)

            # update rule
            self.w = self.w - learning_rate*self.__get_gradient(X[random_idx], y[random_idx])

            ypred = self.predict(X)
            loss = self.loss(y, ypred)
            r2_score = self.r_squared(y, ypred)
            losses.append(loss)
            print(f"epoch: {i}, loss: {loss}, r2: {r2_score}") 
        return losses


    def fit(self, X, y, method="batch", learning_rate=0.001, epochs=300, **kwargs):
        """ Training the model"""
        X = X.copy()
        ones_column = np.ones((len(X),1))
        X = np.concatenate([ones_column, X], axis=1)

        if method == "batch":
            batch_size = X.shape[0] # all the samples

        elif method == "mini-batch":
            if kwargs.get('batch_size') == None:
                batch_size = int(X.shape[0]*0.25)
            else:
                batch_size = kwargs['batch_size']

        elif method == 'stochastic':
            batch_size = 1

        return self.__gradient_descent(X, y, learning_rate, epochs, batch_size)
        
    def predict(self, X):
        return X.dot(self.w)

    def loss(self, y, ypred):
        return ((y - ypred)**2).sum()

    def r_squared(self, ytrue, ypred):
        e_method = ((ytrue-ypred)**2).sum() # sum of squares of residuals
        e_baseline = ((ytrue-ytrue.mean())**2).sum() # total sum of squares
        return 1 - e_method/e_baseline