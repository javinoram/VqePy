import pennylane as qml
from pennylane import numpy as np
from .base import *

class upccgsd_ansatz(base_ansatz):
    sz = 0
    begin_state = None

    
    def set_state(self, electrons, sz):
        self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)
        self.sz = sz


    def circuit(self, theta, obs):
        theta = theta.reshape( qml.kUpCCGSD.shape(k=self.repetition, 
            n_wires=self.qubits, delta_sz=self.sz) )
        
        qml.kUpCCGSD(theta, wires=range(self.qubits),
            k=self.repetition, delta_sz=0, init_state=self.begin_state)
        return qml.expval(obs)