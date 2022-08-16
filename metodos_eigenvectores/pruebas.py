from re import M
from tkinter import E
import numpy as np
import metodos_eigenvectores as met_eigen

# Pruebas de los métodos
diag = np.trunc(10 * np.random.rand(3, 3))
diag = np.diag(np.diag(diag))
V = np.trunc(5 * np.random.rand(3, 3))
A = V@diag@np.linalg.inv(V)
eigenvalores = np.diag(diag)
print(eigenvalores)
print(met_eigen.metodoPotencia(A,np.array([1,1,1]),1000,10**-6))
print(np.linalg.eig(A))