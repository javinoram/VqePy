import pennylane as qml
import math
import scipy.linalg as la
import scipy as sc
from pennylane import numpy as np
from quantumsim.functions.constans import *


'''
Minimization functions in scipy for each variational methods
'''
def scipy_minimization(cost_function, theta, params):
    xs = sc.optimize.minimize(cost_function, theta, method=params["method"],)['x']
    result = cost_function(xs)
    return result, xs

'''
Metodos de minimizacion por gradiente para los algoritmos variacionales


Funcion: gradiend_method_OS
input:
    cost_function: funcion de coste
    theta: vector de parametros de rotacion
    x: vector de posiciones iniciales
    params: parametros de ejecucion
    grad: gradiente del vector x

return:
    energy: lista de valores de energias en cada iteracion
    theta: vector de parametros de rotacion optimos
    x: vector de posiciones de los atomos
'''
def gradiend_method_OS(cost_function, theta, x, params, grad):
    energy = []
    opt_theta = qml.GradientDescentOptimizer(stepsize=params["step_theta"])
    opt_x = qml.GradientDescentOptimizer(stepsize=params["step_x"])

    for _ in range(params["itermax"]):
        theta.requires_grad = True
        x.requires_grad = False
        theta, _ = opt_theta.step(cost_function, theta, x)

        x.requires_grad = True
        theta.requires_grad = False
        _, x = opt_x.step(cost_function, theta, x, grad_fn=grad)

        energy.append(cost_function(theta, x))
        if np.max(grad(theta, x)) <= params["tol"]:
            break

    return energy, theta, x



'''
Funcion: gradiend_method_VQE
input:
    cost_function: funcion de coste
    theta: vector de parametros de rotacion
    params: parametros de ejecucion

return:
    energy: lista de convergencia de energias
    theta: vector de parametros de rotacion optimos
'''
def gradiend_method_VQE(cost_function, theta, params):
    energy = [cost_function(theta)]
    opt_theta = qml.GradientDescentOptimizer(stepsize=params["step_theta"])

    for _ in range(params["itermax"]):
        theta.requires_grad = True
        theta = opt_theta.step(cost_function, theta)
        energy.append(cost_function(theta))
        prev_energy = energy[len(energy)-2]

        conv = np.abs(energy[-1] - prev_energy)
        if conv <= params["tol"]:
            break
    return energy, theta
