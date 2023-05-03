import math
import itertools
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
from classes.global_func import *

class ansatz():
    '''For Spin hamiltonians'''
    repetition: int = 0
    ansatz_pattern: str = ""
    number_params: list = []
    rotation_set: list = []

    '''For Electronic hamiltonians'''
    hf_state: list = None
    singles: list = []
    doubles: list = []

    def __init__(self) -> None:
        pass

    def init_ansatz(self, params, qubits, type_hamiltonian) -> None:
        if type_hamiltonian == "electronic":
            self.repetition = params['repetition']
            self.singles, self.doubles = qml.qchem.excitations(params['electrons'], qubits)
            self.number_params = (len(self.singles) + len(self.doubles))*self.repetition
        else:
            self.repetition = params['repetition']
            self.ansatz_pattern = params['ansatz_pattern']
            self.rotation_set = params['rotation_set']
            self.number_params = [number_rotation_params(self.rotation_set, qubits, self.repetition), 
                                  number_nonlocal_params(self.ansatz_pattern, qubits, self.repetition)]
        return 
    

    '''Given ansatz functions'''
    def simple_given_rotations(self, params):
        if len(self.singles) == 0: return
        for i, term in enumerate(self.singles):
            qml.SingleExcitation(params[i], wires=term)
        return
    
    def double_given_rotations(self, params):
        if len(self.doubles) == 0: return
        for i, term in enumerate(self.doubles):
            qml.DoubleExcitation(params[i], wires=term)
        return
    
    def given_circuit(self, qubits, params, hamiltonian, init_state=None):
        qml.BasisState(init_state, wires=range(qubits))
        for i in range(0, self.repetition):
            self.simple_given_rotations( params[0] )
            self.double_given_rotations( params[1] )
        return qml.expval(hamiltonian)


    '''Spin ansatz functions'''
    def single_rotation(self, params, qubits, correction):
        for i in range( 0, qubits):
            for j in range(correction):
                qml.RZ(params[i][0], wires=[correction*i+j])
                qml.RY(params[i][1], wires=[correction*i+j])
                qml.RX(params[i][2], wires=[correction*i+j])
        return

    def non_local_gates(self, params, qubits, correction):
        if self.ansatz_pattern == 'chain':
            for i in range(0, qubits-1):
                for j in range(correction):
                    qml.CRX(params[i], [correction*i+j, correction*(i+1)+j])
        if self.ansatz_pattern == 'ring':
            if qubits == 2:
                for j in range(correction):
                    qml.CRX(params[i], [j, correction+j])
            else:
                for i in range(0, qubits-1):
                    for j in range(correction):
                        qml.CRX(params[i], [correction*i+j, correction*(i+1)+j])
        return

    def spin_circuit(self, qubits, correction, params, wire, init_state=None, system_object=None):
        qml.BasisState(init_state, wires=range(correction*qubits))

        for i in range(0, self.repetition):
            self.single_rotation(params[0][i], qubits, correction)
            self.non_local_gates(params[1][i], qubits, correction)

        aux = []
        for w in wire:
            for i in range(correction):
                aux.append( correction*w + i)
        return qml.counts(wires=aux)