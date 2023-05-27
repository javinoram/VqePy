from quantumsim.functions.ansatz import *
from quantumsim.functions.constans import *
from pennylane import qchem
import math
import itertools

class variational_quantum_thermalizer_spin(spin05_ansatz):
    hamiltonian_object = None
    hamiltonian_index = []

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
        for term in params['pauli_string']:
            aux = []
            for i, string in enumerate(term[0]):
                if string != 'I': aux.append(i)
            
            if len(aux) != 2:
                raise Exception("Terminos del hamiltoniano tienen mas de 2 interacciones")
            
            self.hamiltonian_index.append(aux)
        self.hamiltonian_object = params['pauli_string']
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
            result_aux = 0
            for i, term in enumerate(self.hamiltonian_index):
                result_term = self.node( theta=ansatz, obs= term, pauli= self.hamiltonian_object[i][0], state= state)
                exchange = self.hamiltonian_object[i][1]
                result_aux += exchange*(result_term[0] - result_term[1] - result_term[2] +result_term[3])
            for j in range(0, len(state)):
                result_aux = result_aux * distribution[j][state[j]]
            result += result_aux
        entropy = calculate_entropy(distribution)
        final_cost = beta * result - entropy
        return final_cost
    
