import math
import itertools
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
from global_func import *

class hamiltonian():
    hamiltonian_object = None
    hamiltonian_index = []
    number_qubits = None
    spin = 0.5

    def __init__(self) -> None:
        pass

    #def init_hamiltonian_file(self, file_name, params) -> None:
    def init_hamiltonian_file(self, params) -> None:
        #symbols, coordinates = qchem.read_structure( file_name )
        hamiltonian, qubits = qchem.molecular_hamiltonian(
            symbols = params['symbols'],
            coordinates = params['coordinates'],
            charge= params['charge'],
            mult= params['mult'],
            basis= params['basis'],
            method= params['method'])
        self.hamiltonian_object = hamiltonian
        self.number_qubits = qubits
        self.spin = 0.5
        pass

    def init_hamiltonian_list(self, params) -> None:
        ham_matrix = {}
        self.number_qubits = len(params['hamiltonian_list'][0][0])
        self.spin = params['spin']
        for term in params['hamiltonian_list']:
            aux = []
            for i, string in enumerate(term[0]):
                if string != 'I': aux.append(i)
            self.hamiltonian_index.append(aux)

        if self.spin == 0.5:
            for term in params['hamiltonian_list']:
                aux = {}
                for i in range(self.number_qubits):
                    aux[i] = term[0][i]
                ham_matrix[qml.pauli.PauliWord(aux)] = term[1] 
            self.hamiltonian_object = qml.pauli.PauliSentence(ham_matrix).hamiltonian([i for i in range(self.number_qubits)])
        else:
            self.hamiltonian_object = params['hamiltonian_list']
        pass