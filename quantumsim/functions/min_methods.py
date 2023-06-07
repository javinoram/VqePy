import pennylane as qml
import math
import scipy.linalg as la
import scipy as sc
from pennylane import numpy as np
from quantumsim.functions.constans import *

'''
Funcion: gradiend_method_VQE
input:
    cost_function: funcion de coste
    theta: vector de parametros de rotacion
    params: parametros de ejecucion

return:
    energy: lista de convergencia de energias
    theta_evol: lista de convergencia de los parametros en la ejecucion
    theta: vector de parametros de rotacion optimos
'''
def gradiend_method_VQE(cost_function, theta, params):
    energy = [cost_function(theta)]
    theta_evol = [theta]
    opt_theta = qml.GradientDescentOptimizer(stepsize=params["step_theta"])

    for _ in range(params["maxiter"]):
        theta.requires_grad = True
        theta = opt_theta.step(cost_function, theta)
        energy.append(cost_function(theta))
        theta_evol.append(theta)
        prev_energy = energy[len(energy)-2]

        conv = np.abs(energy[-1] - prev_energy)
        if conv <= params["tol"]:
            break
    return energy, theta_evol, theta

'''
Funcion: scipy_method_VQE
input:
    cost_function: funcion de coste
    theta: vector de parametros de rotacion
    params: parametros de ejecucion

return:
    energy: lista de convergencia de energias
    theta_evol: lista de convergencia de los parametros en la ejecucion
    theta: vector de parametros de rotacion optimos
'''

def scipy_method_VQE(cost_function, theta, params):
    energy = []
    theta_evol = []

    def cost_aux(x): 
        result = cost_function(x)
        energy.append(result)
        theta_evol.append(x)
        return result
    
    ops = {'maxiter': params["maxiter"]}
    theta = sc.optimize.minimize(cost_aux, theta, method=params["type"], tol=params["tol"], options= ops)['x']
    return energy, theta_evol, theta

'''
Funcion: scipy_method_OS
input:
    cost_function: funcion de coste
    theta: vector de parametros de rotacion
    x: vector de posiciones iniciales
    params: parametros de ejecucion

return:
    energy: lista de convergencia de energias
    theta_evol: lista de convergencia de los parametros en la ejecucion
    theta: vector de parametros optimos
'''
def scipy_method_OS(cost_function, theta, x, params):
    energy = []
    theta_evol = []

    def cost_aux(psi): 
        result = cost_function(psi[len(x):], psi[:len(x)])
        energy.append(result)
        theta_evol.append(psi)
        return result
    
    ops = {'maxiter': params["maxiter"]}
    theta = sc.optimize.minimize(cost_aux, np.concatenate((x,theta), axis=0), method=params["type"], tol=params["tol"], options= ops)['x']
    return energy, theta_evol, theta

'''
Funcion: gradiend_method_OS
input:
    cost_function: funcion de coste
    theta: vector de parametros de rotacion
    x: vector de posiciones iniciales
    params: parametros de ejecucion
    grad: gradiente del vector x

return:
    energy: lista de convergencia de energias
    theta_evol: lista de convergencia de los parametros en la ejecucion
    theta: vector de parametros optimos
'''
def gradiend_method_OS(cost_function, theta, x, params, grad):
    energy = []
    theta_evol = []
    opt_theta = qml.GradientDescentOptimizer(stepsize=params["step_theta"])
    opt_x = qml.GradientDescentOptimizer(stepsize=params["step_x"])

    for _ in range(params["maxiter"]):
        theta.requires_grad = True
        x.requires_grad = False
        theta, _ = opt_theta.step(cost_function, theta, x)

        x.requires_grad = True
        theta.requires_grad = False
        _, x = opt_x.step(cost_function, theta, x, grad_fn=grad)

        theta_evol.append( np.concatenate((x,theta), axis=0) )
        energy.append(cost_function(theta, x))
        if np.max(grad(theta, x)) <= params["tol"]:
            break

    return energy, theta_evol, np.concatenate((x,theta), axis=0)