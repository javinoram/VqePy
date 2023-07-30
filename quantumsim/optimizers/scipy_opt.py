import pennylane as qml
import math
import scipy.linalg as la
import scipy as sc
import warnings
from pennylane import numpy as np
from quantumsim.optimizers import *

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

        if 'tol' in params:
            self.tol = params["tol"]
        
        if 'maxiter' in params:
            self.maxiter = params["maxiter"]
        
        if 'type' in params:
            self.type_method = params["type"]

    def callback(self, x):
        self.nit += 1
        if self.nit == self.maxiter:
            print("Maximo numero de iteraciones")
            warnings.warn("Terminating optimization: iteration limit reached", TookTooManyIters)


    
    def VQE(self, cost_function):
        energy = []
        theta_evol = []
        self.nit = 0

        def cost_aux(x): 
            result = cost_function(x)
            energy.append(result)
            theta_evol.append(x)
            return result
        
        ops = {'maxiter': self.maxiter}
        theta = np.random.random( size=self.number )
        theta = sc.optimize.minimize(cost_aux, theta, method=self.type_method, callback=self.callback, tol=self.tol, options=ops)['x']
        return energy, theta
    

    
    def OS(self, cost_function, x):
        energy = []
        theta_evol = []
        self.nit = 0

        def cost_aux(psi): 
            result = cost_function(psi[len(x):], psi[:len(x)])
            energy.append(result)
            theta_evol.append(psi)
            return result
        
        ops = {'maxiter': self.maxiter}
        theta = theta = np.random.random( size=self.number )
        theta = sc.optimize.minimize(cost_aux, np.concatenate((x,theta), axis=0), method=self.type_method, 
                callback=self.callback, tol=self.tol, options=ops)['x']
        return energy, theta
    

    def VQD(self, cost_function, overlap_cost_function, k):
        energy = []
        previous_theta = []

        for i in range(k):  
            print("state ", i+1)

            def cost_aux(x): 
                result = cost_function(x) 
                for j, previous in enumerate(previous_theta):
                    result += 10*overlap_cost_function(x, previous)
                return result
        
            self.nit = 0
            theta = np.random.random( size=self.number )
            ops = {'maxiter': self.maxiter}
            xs = sc.optimize.minimize(cost_aux, theta, method=self.type_method,
                    callback=self.callback, tol=self.tol, options=ops)['x']
            energy.append( cost_function(xs) )
            previous_theta.append(xs)
        return energy, previous_theta


    def Thermal(self, cost_function, qubits, T):
        energy = []
        theta_evol = []
        self.nit = 0

        constrains = [{'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}]
        for l in range(2**qubits):
            constrains.append( {'type': 'ineq', 'fun': lambda x: 1 - x[l]} )
            constrains.append( {'type': 'ineq', 'fun': lambda x: x[l]} )

        def cost_aux(x): 
            result = cost_function(x, 1.0/T)
            energy.append(result)
            theta_evol.append(x)
            return result
        
        ops = {'maxiter': self.maxiter}
        theta = np.array( [1.0/(2**qubits)  for _ in range(2**qubits)], requires_grad=True)
        theta = sc.optimize.minimize(cost_aux, theta, method=self.type_method, 
                constraints=constrains, callback=self.callback, tol=self.tol, options=ops)['x']
        return energy, theta

