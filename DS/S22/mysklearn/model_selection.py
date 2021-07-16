import numpy as np

def train_test_split(X, y, test_ratio=0.2, shuffle=True):
    test_samples = int(X.shape[0]*test_ratio) # np. of samples in test data
    if shuffle:
        idx = np.arange(0, len(X))
        np.random.shuffle(idx)
        test_idx, train_idx = idx[:test_samples], idx[test_samples:]
        X_train, y_train, X_test, y_test =  X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        return X_train, y_train, X_test, y_test
    else:
        X_test, y_test, X_train, y_train = X[0:test_samples], y[:test_samples], X[test_samples:], y[test_samples:]
        return X_train, y_train, X_test, y_test