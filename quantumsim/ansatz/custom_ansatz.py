import pennylane as qml
from pennylane import numpy as np
from .base import *

class custom_ansatz(base_ansatz):
    begin_state = None
    excitations = []
    
    #def set_state(self, electrons):
    #    self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)
    def set_state(self, state):
        self.begin_state = state

    def set_gates(self, singles=None, doubles=None):
        self.excitations = []
        if doubles != None:
            for term in doubles:
                self.excitations.append(term)

        if singles != None:
            for term in singles:
                self.excitations.append(term)
        return
    

    def circuit(self, theta, obs):
        #qml.BasisState(self.begin_state, wires=range(self.qubits))
        qml.StatePrep(self.begin_state, wires=range(self.qubits))

        for i, gate in enumerate(self.excitations):
            if len(gate) == 4:
                qml.DoubleExcitation(theta[i], wires=gate)
            elif len(gate) == 2:
                qml.SingleExcitation(theta[i], wires=gate)

        return qml.expval(obs)
    