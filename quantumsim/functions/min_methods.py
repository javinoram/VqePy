import pennylane as qml
import math
import scipy.linalg as la
import scipy as sc
import warnings
from pennylane import numpy as np
from quantumsim.functions.funciones import *

class TookTooManyIters(Warning):
    pass

class scipy_optimizer():
    maxiter = 100
    type_method = "SLSQP"
    tol = 1e-6
    nit = 0
    number = 0

    def __init__(self, params):
        self.number = params["number"]

        if params["tol"]:
            self.tol = params["tol"]
        
        if params["maxiter"]:
            self.maxiter = params["maxiter"]
        
        if params["type"]:
            self.type_method = params["type"]


    def callback(self, x):
        self.nit += 1
        if self.nit == self.maxiter:
            print("Maximo numero de iteraciones")
            warnings.warn("Terminating optimization: iteration limit reached", TookTooManyIters)

    def VQD(self, cost_function, overlap_cost_function, k,):
        energy = []
        previous_theta = []

        def cost_aux(x): 
            result = cost_function(x) 
            for previous in previous_theta:
                result += 3*overlap_cost_function(x, previous)
            return result
        
        for _ in range(k):  
            self.nit = 0
            theta = np.array( [np.random.randint(314)/100.0  for _ in range(self.number)], requires_grad=True)
            ops = {'maxiter': self.maxiter}
            xs = sc.optimize.minimize(cost_aux, theta, method=self.type_method, callback=self.callback, tol=self.tol, options=ops)['x']
            energy.append(cost_function(xs))
            previous_theta.append(xs)
        return energy, previous_theta
    
    def VQE(self, cost_function, theta):
        energy = []
        theta_evol = []
        self.nit = 0

        def cost_aux(x): 
            result = cost_function(x)
            energy.append(result)
            theta_evol.append(x)
            return result
        
        ops = {'maxiter': self.maxiter}
        theta = sc.optimize.minimize(cost_aux, theta, method=self.type_method, callback=self.callback, tol=self.tol, options=ops)['x']
        return energy, theta_evol, theta
    
    def OS(self, cost_function, x):
        energy = []
        theta_evol = []
        self.nit = 0

        def cost_aux(psi): 
            result = cost_function(psi[len(x):], psi[:len(x)])
            energy.append(result)
            theta_evol.append(psi)
            print(result)
            return result
        
        ops = {'maxiter': self.maxiter}
        theta = np.array( [np.random.randint(314)/100.0  for _ in range(self.number)], requires_grad=True)
        theta = sc.optimize.minimize(cost_aux, np.concatenate((x,theta), axis=0), method=self.type_method, callback=self.callback, tol=self.tol, options=ops)['x']
        return energy, theta_evol, theta



class gradiend_optimizer():
    maxiter = 100
    theta_optimizer = None
    x_optimizer = None
    tol = 1e-6
    number = 0

    def __init__(self, params):
        self.number = params["number"]
        if params["tol"]:
            self.tol = params["tol"]
        
        if params["maxiter"]:
            self.maxiter = params["maxiter"]

        if params["step_theta"]:
            self.theta_optimizer = qml.GradientDescentOptimizer(stepsize=params["step_theta"])

        if params["step_x"]:
            self.x_optimizer = qml.GradientDescentOptimizer(stepsize=params["step_x"])

    def VQE(self, cost_function, theta):
        energy = [cost_function(theta)]
        theta_evol = [theta]

        for _ in range(self.maxiter):
            theta.requires_grad = True
            theta = self.theta_optimizer.step(cost_function, theta)
            energy.append(cost_function(theta))
            theta_evol.append(theta)
            prev_energy = energy[len(energy)-2]

            conv = np.abs(energy[-1] - prev_energy)
            if conv <= self.tol:
                break
        return energy, theta_evol, theta
        
    
    def OS(self, cost_function, theta, x, grad):
        energy = []
        theta_evol = []

        for _ in range(self.maxiter):
            theta.requires_grad = True
            x.requires_grad = False
            theta, _ = self.theta_optimizer.step(cost_function, theta, x)

            x.requires_grad = True
            theta.requires_grad = False
            _, x = self.x_optimizer.step(cost_function, theta, x, grad_fn=grad)

            if np.max(grad(theta, x)) <= self.tol:
                break

        return energy, theta_evol, np.concatenate((x,theta), axis=0)