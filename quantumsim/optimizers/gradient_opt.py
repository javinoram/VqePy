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

    def VQE(self, cost_function, qubits):

        #theta = np.array( [0.0  for _ in range(self.number)], requires_grad=True)
        theta = np.array( [np.random.randint(314)/100.0  for _ in range(self.number)], requires_grad=True)
        #state =  qml.qchem.hf_state(int(qubits/2), qubits)
        state = [0 for i in range(qubits)]
        energy = [cost_function(theta, state)]
        theta_evol = [theta]

        def cost_aux(x): 
            result = cost_function(x, state) 
            return result

        for _ in range(self.maxiter):
            theta.requires_grad = True
            theta = self.theta_optimizer.step(cost_aux, theta)
            energy.append(cost_function(theta, state))
            theta_evol.append(theta)
            prev_energy = energy[len(energy)-2]

            conv = np.abs(energy[-1] - prev_energy)
            if conv <= self.tol:
                break
        return energy, theta
           
    def OS(self, cost_function, x, grad):
        energy = []
        theta = np.array( [0.0  for _ in range(self.number)], requires_grad=True)

        for _ in range(self.maxiter):
            theta.requires_grad = True
            x.requires_grad = False
            theta, _ = self.theta_optimizer.step(cost_function, theta, x)

            x.requires_grad = True
            theta.requires_grad = False
            _, x = self.x_optimizer.step(cost_function, theta, x, grad_fn=grad)

            if np.max(grad(theta, x)) <= self.tol:
                break

        return energy, np.concatenate((x,theta), axis=0)
    
    def VQD(self, cost_function, overlap_cost_function, k, qubits):
        previous_theta = []
        energy_final = [] 

        combos = itertools.product([0, 1], repeat=qubits)
        s = [list(c) for c in combos]

        for i in range(k):

            def cost_aux(x): 
                result = cost_function(x, s[i]) 
                for previous in previous_theta:
                    result += 5*overlap_cost_function(x, previous, s[i])
                return result
        
            print("state ", i+1)
            self.nit = 0
            #theta = np.array( [np.random.randint(314)/100.0  for _ in range(self.number)], requires_grad=True)
            theta = np.array( [0.0  for _ in range(self.number)], requires_grad=True)
            energy = [cost_function(theta, s[i])]
            theta_evol = [theta]
            for _ in range(self.maxiter):
                theta.requires_grad = True
                theta = self.theta_optimizer.step(cost_aux, theta)
                energy.append(cost_function(theta, s[i]))
                theta_evol.append(theta)
                prev_energy = energy[len(energy)-2]

                conv = np.abs(energy[-1] - prev_energy)
                if conv <= self.tol:
                    break
            energy_final.append( energy[-1] )
            previous_theta.append( theta )
        return energy_final, theta