import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from mpl_toolkits import mplot3d


# Training phase/ Training the LR model/ Find optimal weights
def fit(X, y):
    """
    X: Feature matrix: (n_samples, n_features)
    y: y_true: (n_samples,1)
    Returns: weights
    weights: optimal weights (n_features, 1)
    """
    X = X.copy()
    ones_column = np.ones((len(X),1))
    X = np.concatenate([ones_column, X], axis=1)

    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w

# prediction
def predict(X, w):
    """
    X: Feature matrix: (n_samples, n_features)
    w: weight vector: (n_fetures, 1)
    Returns:
    y: y_pred = X.w (n_samples,1)
    """
    X = X.copy()
    ones_column = np.ones((len(X),1))
    X = np.concatenate([ones_column, X], axis=1)

    return X.dot(w)

# r_squared
def r_squared(ytrue, ypred):
    e_method = ((ytrue-ypred)**2).sum() # sum of squares of residuals
    e_baseline = ((ytrue-ytrue.mean())**2).sum() # total sum of squares
    return 1 - e_method/e_baseline

# loss function
def loss(ytrue, ypred):
    return ((ytrue-ypred)**2).sum()

X, y, coeff =  make_regression(n_samples=100, n_features=2, coef=True, noise=0.5, bias=3, random_state=70)
# print(X.shape, y.shape)

# Train the model/ learn the optimal weights
w = fit(X, y)

####################################################

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')

ax.scatter(X[:,0], X[:,1], y, c=y, cmap='seismic')


f1 = np.linspace(X[:,0].min(), X[:,0].max(), 50)
f2 = np.linspace(X[:,1].min(), X[:,1].max(), 50)
f1, f2 = np.meshgrid(f1, f2)

# prediction plane
X_ = np.concatenate([f1.reshape(-1,1), f2.reshape(-1,1)], axis=1)
pred = predict(X_, w).reshape(f1.shape)
ax.plot_surface(f1, f2, pred, alpha=0.5, cmap='seismic')

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Output (y)")

plt.show()