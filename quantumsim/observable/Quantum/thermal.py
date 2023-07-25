from pennylane import numpy as np
from quantumsim.ansatz import *
import pandas as pd
import pennylane as qml


class quantum_thermal(HE_ansatz_thermal):
    dtype = "float64"
    boltz = 8.617333262e-5 #eV/K

    energy = None
    states = None
    parity_terms = None

    def __init__(self, params):
        self.qubits = params["sites"]
        self.energy = np.array(params["energy"])
        self.states = params["states"]
        self.parity_terms = np.array([ parity(i, 0.5, self.qubits) for i in range(2**(self.qubits*self.correction)) ]) 
        return
        
    def expected_value(self):
        expval = []
        for i,state in enumerate(self.states):
            result_prob = self.node(theta = self.states[i], state=[0 for j in range(self.qubits)])
            expval.append( np.sum(result_prob*self.parity_terms[:result_prob.shape[0]]) )
        return np.array(expval)
    
    def entropy(self, dist, epsilon=1e-8):
        result = 0.0
        for i in dist:
            i = max(i, epsilon)  # Ensure the probability is at least epsilon
            result += -i * np.log(i)
        return result


    def cost_function(self, dist, beta):
        #expval = self.expected_value()
        #entropy = np.sum(-dist*np.log(dist))
        entropy = self.entropy(dist)
        result = np.sum( self.energy*dist )
        final_cost = beta * result - entropy
        return final_cost