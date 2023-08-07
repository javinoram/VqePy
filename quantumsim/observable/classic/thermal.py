from pennylane import numpy as np
from quantumsim.ansatz import *
import pandas as pd
import pennylane as qml


class hybrid_thermal():
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

    ###Calculadora de valores esperados
    def expected_value(self, op):
        expval = []

        if op == "enthalpy":
            return self.energy
        elif op == "magnetization":
            for i,state in enumerate(self.states):
                result_prob = self.node(theta = self.states[i])
                expval.append( np.sum(result_prob*self.parity_terms[:result_prob.shape[0]]) )
            return np.array(expval)
        else:
            return None


    ###Estimador de la distribucion de probabilidades
    def distribution(self, T):
        return np.exp(np.divide(-self.energy, T*self.boltz, dtype=self.dtype), dtype=self.dtype)
    
    