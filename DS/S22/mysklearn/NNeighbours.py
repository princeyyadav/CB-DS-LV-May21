import numpy as np

class KNN:

    def predict(self, X, y, test_point, k=3):
        # compute distance
        distance = ((X - test_point)**2).sum(axis=1)**0.5
        knn_indexes = np.argsort(distance)[:k] # indices of k nearest neighbour

        # get the category of my NN
        knn_cat = y[knn_indexes]
        cls, count = np.unique(knn_cat, return_counts=True)
        # d = {key:v for key, v in zip(cls, count)}
        # print(d)

        max_count_idx = np.argmax(count)
        pred_cat = cls[max_count_idx] # category with maximum count

        return pred_cat

    def accuracy(self, ypred, ytrue):
        return (ypred==ytrue).mean()