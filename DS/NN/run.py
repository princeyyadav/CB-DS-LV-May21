from activation import Sigmoid
from loss import BinaryCrossEntropy
from optimizer import GradientDescentOptimizer
from layer import Dense
from model import Sequential
from sklearn.datasets import make_gaussian_quantiles


if __name__ == '__main__':
    X, y = make_gaussian_quantiles(n_samples=200,n_classes=2)
    y = y.reshape(-1,1)
    print("X", X.shape, "y:", y.shape)

    model = Sequential(BinaryCrossEntropy())
    model.add(Dense(input_size = 2, activation=Sigmoid(), units=3))
    model.add(Dense(input_size = 3, activation=Sigmoid(), units=2))
    model.add(Dense(input_size = 2, activation=Sigmoid(), units=1))

    model.summary()

    ypred = model.predict(X)
    print("Ypred", ypred.shape)
    print("Loss", model.loss(y, ypred))
    print("Accuracy", model.accuracy(y, ypred))

    model.fit(X, y, epochs=1000, optimizer=GradientDescentOptimizer, learning_rate=0.005, verbose=1)
