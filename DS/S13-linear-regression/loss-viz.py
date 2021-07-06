import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def loss(m=29.97, c=-4):
    return ((2 - (1*m + c))**2)

M = np.linspace(-50, 50, 50)
C = np.linspace(-50, 50, 50)

M, C = np.meshgrid(M, C)
print(M.shape, C.shape)

# ypred = predict(X)
# print(ypred.shape)

L = loss(M.reshape(-1,1), C.reshape(-1,1))
L = L.reshape(-1, 50)
print(L.shape)

fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')

ax.plot_surface(M, C, L, cmap='seismic', alpha=0.5)
ax.set_xlabel('M')
ax.set_ylabel('C')
ax.set_zlabel('Loss')

plt.show()