from quantumsim.variational_quantum_eigensolver import *
from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
import sys
import time

print(sys.argv)

symbols = ["H", "H", "H", "H", "H", "H", "H", "H",]
coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0,
        0.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0,], requires_grad=True)
params_ansatz = {'repetition': 1,'electrons': 8,}
params_alg = {'backend': "default.qubit", 'interface':"autograd"}



object_vqe = variational_quantum_eigensolver_electronic(symbols, coordinates)
object_vqe.set_device(params_alg)
object_vqe.set_hiperparams_circuit(params_ansatz)
object_vqe.set_node(params_alg)
number = len(object_vqe.singles) + len(object_vqe.doubles)

print("Ground state")
print(number)
theta = np.array([0.0  for _ in range(number)])
result = object_vqe.cost_function(theta)
print(result)

print(object_vqe.draw_circuit)

seconds = time.time()
xs = sc.optimize.minimize(object_vqe.cost_function, theta, method="COBYLA",)['x']
result = object_vqe.cost_function(xs)
seconds2 = time.time()
print(result)
print(xs)
print(seconds2- seconds)