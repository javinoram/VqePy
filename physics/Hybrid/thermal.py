from pennylane import numpy as np
from physics.functions.ansatz import *
import pandas as pd
import pennylane as qml


class hybrid_thermal(HE_ansatz):
    dtype = "float64"
    boltz = 8.617333262e-5 #eV/K

    energy = None
    states = None
    parity_terms = None

    def __init__(self, params):
        self.qubits = params["sites"]
        self.energy = params["energy"]
        self.states = params["states"]
        self.dtype = params["dtype"]
        self.parity_terms = np.array([ parity(i, 0.5, self.qubits) for i in range(2**(self.qubits*self.correction)) ]) 
        return

    def distribution(self, T):
        return np.exp(np.divide(-self.energy, T*self.boltz, dtype=self.dtype), dtype=self.dtype)

    def expected_value(self, op, thetas):
        expval = []
        
        if op == "enthalpy":
            return self.energy
        
        elif op == "magnetization":
            for i,state in enumerate(self.states):
                result_prob = self.node(theta = thetas[i], state= state)
                expval.append( np.sum(result_prob*self.parity_terms[:result_prob.shape[0]]) )
            return np.array(expval)
        else:
            return None


    #def Specific_heat(self, t):
    #    partition = np.exp(np.divide(-self.energy, t*self.boltz, dtype=self.dtype), dtype=self.dtype)

    #    aux1 = np.divide(np.sum(partition*self.energy), np.sum(partition))
    #    aux2 = np.divide(np.sum(partition*(self.energy**2)), np.sum(partition))

    #    return np.divide(aux2- (aux1**2), (t*t*self.boltz), dtype=self.dtype)
    
    