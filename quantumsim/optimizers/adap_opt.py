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


'''
The adaptative optimizer use a pool of uccsd operators (single and double operators)
This optimizer returns the minimum circuit's depth with the minimun number
of parameters.

'''
class adap_optimizer():
    maxiter = 20
    optimizer = None
    operator_pool = None

    def __init__(self, params):
        
        if 'maxiter' in params:
            self.maxiter = params["maxiter"]

        if 'theta' in params:
            self.optimizer = qml.GradientDescentOptimizer(stepsize=params['theta'][1])
        
        if 'electrons' in params and 'qubits' in params:
            singles, doubles = qml.qchem.excitations(params['electrons'], params['qubits'])
            print(len(singles)+len(doubles))
            self.operator_pool = [ doubles, singles ]




    def MinimumCircuit(self, cost_fn):
        params_doubles = [0.0]*len(self.operator_pool[0])
        circuit_gradient = qml.grad(cost_fn, argnum=0)
        grad_doubles = circuit_gradient(params_doubles, excitations=self.operator_pool[0])
        doubles_select = [self.operator_pool[0][i] for i in range(len(self.operator_pool[0])) if abs(grad_doubles[i]) > 1.0e-5]
        params_doubles = np.zeros(len(doubles_select), requires_grad=True)
        if len(params_doubles) != 0:
            for n in range(self.maxiter):
                params_doubles = self.optimizer.step(cost_fn, params_doubles, excitations=doubles_select)
        print( self.operator_pool[0] )
        print( grad_doubles )


        circuit_gradient = qml.grad(cost_fn, argnum=0)
        params_single = [0.0] * len(self.operator_pool[1])   
        grad_singles = circuit_gradient(
            params_single,
            excitations=self.operator_pool[1],
            gates_select=doubles_select,
            params_select=params_doubles
        )
        singles_select = [self.operator_pool[1][i] for i in range(len(self.operator_pool[1])) if abs(grad_singles[i]) > 1.0e-5]
        return singles_select, doubles_select
    