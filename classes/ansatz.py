import math
import itertools
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
from global_func import *

class ansatz():
    repetition: int = 0
    ansatz_pattern: str = ""
    number_rotations: int = 0
    number_nonlocal: int = 0
    rotation_set: list = []

    def __init__(self) -> None:
        pass

    def init_ansatz(self, params, qubits) -> None:
        self.repetition = params['repetition']
        self.ansatz_pattern = params['ansatz_pattern']
        self.rotation_set = params['rotation_set']
        self.number_rotations = number_rotation_params(self.rotation_set, qubits, self.repetition)
        self.number_nonlocal = number_nonlocal_params(self.ansatz_pattern, qubits, self.repetition)
        return 

    def single_rotation(self, phi_params, qubits, spin):
        phi_params = np.array(phi_params).T
        correction = math.ceil( (int( 2*spin+1 ))/2  )
        for i in range( 0, qubits):
            for j in range(correction):
                qml.RZ(phi_params[i][0], wires=[correction*i+j])
                qml.RY(phi_params[i][1], wires=[correction*i+j])
                qml.RX(phi_params[i][2], wires=[correction*i+j])
        return

    def non_local_gates(self, phi_params, qubits, spin):
        correction = math.ceil( (int( 2*spin+1 ))/2  )

        if self.ansatz_pattern == 'chain':
            for i in range(0, qubits-1):
                for j in range(correction):
                    qml.CRX(phi_params[i], [correction*i+j, correction*(i+1)+j])
        if self.ansatz_pattern == 'ring':
            if qubits == 2:
                for j in range(correction):
                    qml.CRX(phi_params[i], [j, correction+j])
            else:
                for i in range(0, qubits-1):
                    for j in range(correction):
                        qml.CRX(phi_params[i], [correction*i+j, correction*(i+1)+j])
        return

    def quantum_circuit(self, qubits, spin,  rotation_params, coupling_params, wire, sample=None, system_object=None):
        correction = math.ceil( (int( 2*spin+1 ))/2  )
        qml.BasisState(sample, wires=range(correction*qubits))
        for i in range(0, self.repetition):
            self.single_rotation(rotation_params[i], qubits, spin)
            #qml.DoubleExcitation(coupling_params[i][0], wires=[0, 1, 2, 3])
            self.non_local_gates(coupling_params[i], qubits, spin)
            #qml.broadcast(
            #    unitary=qml.CRX, pattern=self.ansatz_pattern,
            #    wires=range(qubits), parameters=coupling_params[i]
            #)

        if system_object == None:
            aux = []
            for w in wire:
                for i in range(correction):
                    aux.append( correction*w + i)
            return qml.counts(wires=aux)
        else:
            return qml.expval(system_object)