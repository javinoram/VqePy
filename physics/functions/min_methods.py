import pennylane as qml
import math
import scipy.linalg as la
import scipy as sc
from pennylane import numpy as np
from quantumsim.functions.constans import *

'''
Funcion: gradiend_method_VQT
input:
    cost_function: funcion de coste
    theta: vector de parametros de rotacion
    dist: vector de parametros de la districion de probabilidad
    beta: parametro de temperatura (1/T)
    params: parametros de ejecucion

return:
    energy: lista de convergencia de energias
    theta_evol: lista de convergencia de los parametros en la ejecucion
    theta: vector de parametros optimos
'''
def gradiend_method_VQT(cost_function, theta, dist, beta, params):
    def cost_aux(psi): 
        result = cost_function(psi, beta)
        return result
    
    theta = np.concatenate((dist, theta), axis=0)
    energy = [cost_aux(theta)]
    theta_evol = [theta]
    opt_theta = qml.GradientDescentOptimizer(stepsize=params["step_theta"])

    for _ in range(params["maxiter"]):
        theta.requires_grad = True
        theta = opt_theta.step(cost_aux, theta)

        energy.append(cost_aux(theta))
        theta_evol.append(theta)
        prev_energy = energy[len(energy)-2]

        conv = np.abs(energy[-1] - prev_energy)
        if conv <= params["tol"]:
            break
    return theta


'''
Funcion: gradiend_method_VQT
input:
    cost_function: funcion de coste
    theta: vector de parametros de rotacion
    dist: vector de parametros de la districion de probabilidad
    beta: parametro de temperatura (1/T)
    params: parametros de ejecucion

return:
    energy: lista de convergencia de energias
    theta_evol: lista de convergencia de los parametros en la ejecucion
    theta: vector de parametros optimos
'''
def scipy_method_VQT(cost_function, theta, dist, beta, params):

    def cost_aux(psi): 
        result = cost_function(psi, beta)
        return result
    
    ops = {'maxiter': params["maxiter"], 'tol': params["tol"]}
    theta = sc.optimize.minimize(cost_aux, np.concatenate((dist, theta), axis=0), method=params["type"], options= ops)['x']
    return theta