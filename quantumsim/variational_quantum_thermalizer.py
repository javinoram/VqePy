from quantumsim.functions.ansatz import *
from quantumsim.functions.constans import *
from pennylane import qchem
import math
import itertools

class variational_quantum_thermalizer_spin(spin05_ansatz):
    hamiltonian_object = None

    '''
    Iniciador de la clase que construye la matriz de indices, todas las variables
    son guardadas en la clase
    input:
        params: dict
    return:
        result: none
    '''
    def __init__(self, params):
        self.qubits = len(params['pauli_string'][0][0])

        dict_pauli = {}
        for term in params['pauli_string']:
            pauli_word = {}
            for i, string in enumerate(term[0]):
                pauli_word[i] = string
            dict_pauli[qml.pauli.PauliWord(pauli_word)] = term[1]

        self.hamiltonian_object = qml.pauli.PauliSentence(dict_pauli)
        return
    

    '''
    Funcion de coste: Funcion de coste usando descomposicion de suma de probabilidades
    input:
        theta: list [float]
    return:
        result: float
    '''
    def cost_function(self, theta, beta):
        ansatz = theta[self.qubits:]
        dist_params = theta[:self.qubits]

        distribution = prob_dist(dist_params)

        combos = itertools.product([0, 1], repeat=self.qubits)
        s = [list(c) for c in combos]
        
        result = 0.0
        for state in s:
            result += self.node( theta=ansatz, obs= self.hamiltonian_object, state= state)
            for j in range(0, len(state)):
                result = result * distribution[j][state[j]]
        entropy = calculate_entropy(distribution)
        final_cost = beta * result - entropy
        return final_cost
    
