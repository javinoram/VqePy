from physics.functions.ansatz import *
from physics.functions.constans import *
from pennylane import qchem
import math
import itertools


class spin_hamiltonian(spin_ansatz):
    hamiltonian_object = None
    hamiltonian_group = None

    def __init__(self, params):
        self.qubits = len(params['pauli_string'][0][1])
        self.spin = params['spin']
        self.correction = math.ceil( (int( 2*self.spin+1 ))/2  )
        self.hamiltonian_object = params['pauli_string']
        self.hamiltonian_group = conmute_group(params['pauli_string'])
        return
    

    def magnetization(self, theta, time, n):
        value_per_site = []
        result_probs = self.node(theta = theta, hamiltonian=self.hamiltonian_object, time=time, n=n)
        for probs in result_probs:
            result = 0.0
            for j in range(len(probs)):
                result += probs[j]*parity(j)
            value_per_site.append(result)
        return value_per_site
    
    def thermal_cost_function(self, theta, beta):
        ansatz = theta[self.qubits:]
        dist_params = theta[:self.qubits]
        distribution = prob_dist(dist_params)
        combos = itertools.product([0, 1], repeat=self.qubits)
        s = [list(c) for c in combos]
        
        result = 0.0
        for state in s:
            result_aux = 0.0
            for group in self.hamiltonian_group:

                #Valor esperado del hamiltoniano
                result_probs = self.node( theta=ansatz, obs= group, state= state)
                for k, probs in enumerate(result_probs):
                    aux = group[k][0]
                    for j in range(len(probs)):
                        result_aux += aux*probs[j]*parity(j)
            
            #Ponderacion termica
            for j in range(0, len(state)):
                result_aux = result_aux * distribution[j][state[j]]
            result += result_aux
        
        #Valor final
        entropy = calculate_entropy(distribution)
        final_cost = beta * result - entropy
        return final_cost

    def specific_heat(self, theta, T):
        ansatz = theta[self.qubits:]
        dist_params = theta[:self.qubits]
        distribution = prob_dist(dist_params)
        combos = itertools.product([0, 1], repeat=self.qubits)
        s = [list(c) for c in combos]
        k_b = 8.617333262e-5 
        energy = 0.0
        energy_square = 0.0

        for state in s:
            result_aux = 0
            result_aux_2 = 0.0

            for group in self.hamiltonian_group:
                #Valor esperado del hamiltoniano
                result_probs = self.node( theta=ansatz, obs= group, state= state)
                for k, probs in enumerate(result_probs):
                    aux = group[k][0]
                    for j in range(len(probs)):
                        result_aux += aux*probs[j]*parity(j)
            result_aux_2 = result_aux*result_aux
            
            for j in range(0, len(state)):
                result_aux = result_aux * np.sqrt(distribution[j][state[j]])
                result_aux_2 = result_aux_2 * np.sqrt(distribution[j][state[j]])
            energy += result_aux
            energy_square += result_aux_2
            #print(energy, energy_square)
        return (energy_square - energy*energy)/(T*T*k_b)
