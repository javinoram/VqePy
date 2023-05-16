from quantumsim.variational_quantum_eigensolver import *
from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
import sys

print(sys.argv)
J = -1
list_h = [(("ZZI", -J)), (("IZZ", -J))]
params = {"list": list_h, "spin":0.5}
params_ansatz = {'repetition': 3,'shots': 2**13,}
params_alg = {'backend': "default.qubit", 'interface':"autograd"}



object_vqe = variational_quantum_eigensolver_spin(params)
object_vqe.set_device(params_alg)
object_vqe.set_hiperparams_circuit(params_ansatz)
object_vqe.set_node(params_alg)


print("Ground state")
number = (len(object_vqe.rotation_set)*object_vqe.qubits + len(object_vqe.nonlocal_set)*(object_vqe.qubits+1))*object_vqe.repetition
theta = np.array([1.0  for _ in range(number)])
result = object_vqe.cost_function(theta)
print(result)

xs = sc.optimize.minimize(object_vqe.cost_function, theta, method="SLSQP",options={"maxiter":1000})['x']
result = object_vqe.cost_function(xs)
print(result)
print(xs)

print("Ground state numpy")
sz = np.array([[1,0],[0,-1]])
si = np.array([[1,0],[0,1]])
h = -J*( np.kron(sz, np.kron(sz,si)) + np.kron(si, np.kron(sz,sz)) )
ee = np.linalg.eigvals(h)
print(ee)