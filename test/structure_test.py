from quantumsim.optimizacion_structure import *
from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
import sys

print(sys.argv)

symbols = ["H", "H", "H"]
coordinates = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0], requires_grad=True)
params_ansatz = {'repetition': 1,'electrons': 2,}
params_alg = {'backend': "default.qubit", 'interface':"autograd"}



object_struc = optimization_structure(symbols, coordinates)
object_struc.set_device(params_alg)
object_struc.set_hiperparams_circuit(params_ansatz)
object_struc.set_node(params_alg)
number = len(object_struc.singles) + len(object_struc.doubles)

print("Structure")
print(number)
theta = np.array([0.0  for _ in range(number)], requires_grad=True)
result = object_struc.cost_function(theta, coordinates)
print(result)

a = object_struc.grad_x(theta, coordinates)
print(a)

opt_theta = qml.GradientDescentOptimizer(stepsize=0.4)
opt_x = qml.GradientDescentOptimizer(stepsize=0.8)

energy = []
bond_length = []
bohr_angs = 0.529177210903
x = coordinates


for n in range(100):
    theta.requires_grad = True
    x.requires_grad = False
    theta, _ = opt_theta.step(object_struc.cost_function, theta, x)

    # Optimize the nuclear coordinates
    x.requires_grad = True
    theta.requires_grad = False
    _, x = opt_x.step(object_struc.cost_function, theta, x, grad_fn=object_struc.grad_x)

    energy.append(object_struc.cost_function(theta, x))
    bond_length.append(np.linalg.norm(x[0:3] - x[3:6]) * bohr_angs)

    if n % 4 == 0:
        print(f"Step = {n},  E = {energy[-1]:.8f} Ha,  bond length = {bond_length[-1]:.5f} A")

    # Check maximum component of the nuclear gradient
    if np.max(object_struc.grad_x(theta, x)) <= 1e-05:
        break

print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
print("\n" "Ground-state equilibrium geometry")
print("%s %4s %8s %8s" % ("symbol", "x", "y", "z"))
for i, atom in enumerate(symbols):
    print(f"  {atom}    {x[3 * i]:.4f}   {x[3 * i + 1]:.4f}   {x[3 * i + 2]:.4f}")