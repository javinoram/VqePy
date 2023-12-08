import pennylane as qml
from pennylane import numpy as np
from .base import *

class uccds_ansatz(base_ansatz):
    begin_state = None
    singles = None
    doubles = None

    def set_exitations(self, electrons, sz):
        singles, doubles = qml.qchem.excitations(electrons, self.qubits, delta_sz=sz)
        self.singles, self.doubles = qml.qchem.excitations_to_wires(singles, doubles)
    

    def set_state(self, electrons, sz):
        self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)


    def circuit(self, theta, obs):
        qml.UCCSD(theta, range(self.qubits), self.singles, self.doubles, self.begin_state)
        return qml.expval(obs)
