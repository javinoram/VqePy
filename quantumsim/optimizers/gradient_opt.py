import pennylane as qml
import math
import scipy.linalg as la
import scipy as sc
import warnings
import itertools
from pennylane import numpy as np
from quantumsim.optimizers import *

class TookTooManyIters(Warning):
    pass


class gradiend_optimizer():
    maxiter = 100
    theta_optimizer = None
    x_optimizer = None
    tensor_metric = None
    tol = 1e-6
    number = 0
    begin_state= None

    def __init__(self, params):
        self.number = params["number"]

        if 'tol' in params:
            self.tol = params["tol"]
        
        if 'maxiter' in params:
            self.maxiter = params["maxiter"]

        if 'theta' in params:
            if params['theta'][0] == "generic":
                self.theta_optimizer = qml.GradientDescentOptimizer(stepsize=params['theta'][1])
            elif params['theta'][0] == "adam":
                self.theta_optimizer = qml.AdamOptimizer(stepsize=params['theta'][1])
            elif params['theta'][0] == "adagrad":
                self.theta_optimizer = qml.AdagradOptimizer(stepsize=params['theta'][1])

        if 'x' in params:
            if params['x'][0] == "generic":
                self.x_optimizer = qml.GradientDescentOptimizer(stepsize=params['x'][1])
            elif params['x'][0] == "adam":
                self.x_optimizer = qml.AdamOptimizer(stepsize=params['x'][1])
            elif params['x'][0] == "adagrad":
                self.x_optimizer = qml.AdagradOptimizer(stepsize=params['x'][1])
        
        self.begin_state = np.random.random( size=self.number )*(np.pi/180.0)



    def VQE(self, cost_function):
        theta = self.begin_state
        energy = [cost_function(theta)]
        theta_evol = [theta]

        for _ in range(self.maxiter):
            theta.requires_grad = True
            print("+1")
            theta = self.theta_optimizer.step(cost_function, theta)
            energy.append(cost_function(theta))
            theta_evol.append(theta)
            prev_energy = energy[len(energy)-2]
            conv = np.abs(energy[-1] - prev_energy)
            if conv <= self.tol:
                break
        return energy, theta
           


    def OS(self, cost_function, x, grad):
        theta = self.begin_state
        for _ in range(self.maxiter):
            theta.requires_grad = True
            x.requires_grad = False
            theta, _ = self.theta_optimizer.step(cost_function, theta, x)

            x.requires_grad = True
            theta.requires_grad = False
            _, x = self.x_optimizer.step(cost_function, theta, x, grad_fn=grad)

            if np.max(grad(theta, x)) <= self.tol:
                break
        return np.concatenate((x,theta), axis=0)
    
    
    def VQD(self, cost_function, overlap_cost_function, k):
        previous_theta = []
        energy_final = [] 

        for i in range(k):
            def cost_aux(x): 
                result = cost_function(x) 
                for previous in previous_theta:
                    result += 10*overlap_cost_function(x, previous)
                return result
        
            print("state ", i+1)
            self.nit = 0

            
            theta = self.begin_state
            energy = [cost_function(theta)]
            theta_evol = [theta]
            for _ in range(self.maxiter):
                theta.requires_grad = True
                theta = self.theta_optimizer.step(cost_aux, theta)
                energy.append(cost_function(theta))
                theta_evol.append(theta)
                prev_energy = energy[len(energy)-2]

                conv = np.abs(energy[-1] - prev_energy)
                if conv <= self.tol:
                    break
            energy_final.append( energy[-1] )
            previous_theta.append( theta )
        return energy_final, theta
    

    def Thermal(self, cost_function, qubits, T):
        energy = []
        theta_evol = []

        bounds = []
        for l in range(2**qubits):
            bounds.append( ((0,1)) )

        def cost_aux(x): 
            result = cost_function(x, 1.0/T)
            if T>=1:
                result += 10*np.abs(1 - np.sum(x))
            else:
                result += np.exp(1.0/T)*np.abs(1 - np.sum(x))
            energy.append(result)
            theta_evol.append(x)
            return result
        
        for _ in range(self.maxiter):
            theta.requires_grad = True
            theta = self.theta_optimizer.step(cost_aux, theta)
            energy.append(cost_function(theta))
            theta_evol.append(theta)
            prev_energy = energy[len(energy)-2]

            conv = np.abs(energy[-1] - prev_energy)
            if conv <= self.tol:
                break
        return energy, theta