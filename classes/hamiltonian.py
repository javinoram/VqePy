import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
from classes.global_func import *

class hamiltonian():
    hamiltonian_type = ""
    hamiltonian_object = None
    hamiltonian_index = []
    number_qubits = None
    spin = 0.5

    def __init__(self) -> None:
        pass

    def init_hamiltonian_file(self, params) -> None:
        if self.hamiltonian_object != None:
            raise "Hamiltonian object already setup"
        else:
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
            self.hamiltonian_type = "electronic"
        return 

    def init_hamiltonian_list(self, params) -> None:
        if self.hamiltonian_object != None:
            raise "Hamiltonian object already setup"
        else:
            self.number_qubits = len(params['hamiltonian_list'][0][0])
            self.spin = params['spin']
            for term in params['hamiltonian_list']:
                aux = []
                for i, string in enumerate(term[0]):
                    if string != 'I': aux.append(i)
                self.hamiltonian_index.append(aux)
            self.hamiltonian_object = params['hamiltonian_list']
            self.hamiltonian_type = "spin"
        return