from classes.hamiltonian import *
from classes.ansatz import *
from classes.global_func import *

import math
import itertools
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc



class variational_quantum_eigensolver(hamiltonian, ansatz):
    dev = None
    node = None
    backend = None
    optimization_method = ""
    optimization_alg_params = {}
    shots = 3000

    def __init__(self, params_hamiltonian, params_ansatz, params_alg) -> None:
        if 'symbols' in params_hamiltonian and 'coordinates' in params_hamiltonian:
            self.init_hamiltonian_file(params_hamiltonian)
            self.init_ansatz(params_ansatz, self.number_qubits, self.hamiltonian_type)
            correction = math.ceil( (int( 2*self.spin+1 ))/2  )
            self.dev = qml.device(params_alg['backend'], wires=correction*self.number_qubits, shots=self.shots)
            self.node = qml.QNode(self.given_circuit, self.dev, interface="autograd")
        else:
            self.init_hamiltonian_list(params_hamiltonian)
            self.init_ansatz(params_ansatz, self.number_qubits, self.hamiltonian_type)
            correction = math.ceil( (int( 2*self.spin+1 ))/2  )
            self.dev = qml.device(params_alg['backend'], wires=correction*self.number_qubits, shots=self.shots)
            self.node = qml.QNode(self.spin_circuit, self.dev, interface="autograd")

        self.optimization_alg_params = params_alg['optimization_alg_params'],
        self.optimization_method = params_alg['optimization_method'],
        pass

    def ground_state_calculation(self, theta: list, electrons: int ):
        correction = math.ceil( (int( 2*self.spin+1 ))/2  )
        hf = qml.qchem.hf_state(electrons*correction, self.number_qubits*correction)

        general_cost_function = lambda theta: self.cost_function_VQE(theta, hf)
        xs = sc.optimize.minimize(general_cost_function, theta, method=self.optimization_method[0],
                                options=self.optimization_alg_params[0])['x']
        return xs, general_cost_function(xs)
    

    def cost_function_VQE(self, theta: list, state: list) -> float:
        if self.hamiltonian_type == "spin":

            ansatz_params_1 = theta[0 :self.number_nonlocal]
            ansatz_params_2 = theta[self.number_nonlocal: ]
            coupling = np.split(ansatz_params_1, self.repetition)
            split = np.split(ansatz_params_2, self.repetition)
            rotation = []
            for s in split:
                rotation.append(np.split(s, 3))


            result= 0.0
            for i, term in enumerate(self.hamiltonian_index):
                result_term = self.node( qubits= self.number_qubits, spin=self.spin, rotation_params = rotation, coupling_params = coupling, wire = term, sample=state )
                for _, dict_term in enumerate( conts_spin[ str(self.spin) ]["2"] ):
                    if dict_term in result_term:
                        exchange = self.hamiltonian_object[i][1]
                        prob = result_term[dict_term]/self.shots
                        const_state = conts_spin[ str(self.spin) ]["2"][dict_term]
                        result += exchange*prob*const_state
        else:
            params_1 = theta[:self.repetition*len(self.singles)]
            params_2 = theta[self.repetition*len(self.singles):]
            result = self.node( qubits= self.number_qubits, params = [params_1, params_2], hamiltonian=self.hamiltonian_object, init_state=state)
            print(result)
        return result