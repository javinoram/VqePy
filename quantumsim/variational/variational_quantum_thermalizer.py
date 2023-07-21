from quantumsim.ansatz import *
from quantumsim.optimizers.funciones import *
import itertools

class vqt_spin(he_ansatz):
    hamiltonian_object = None
    groups_caractericts = None
    coeff_object = None
    parity_terms = None

    def __init__(self, params):
        self.qubits = params['sites']
        self.spin = 0.5
        self.correction = 1
        terms = []
        coeff = []
        self.hamiltonian_object = None

        if params["pattern"] in ("open", "close"):
            for i in range(self.qubits-1):
                Xterm = ["I"]*self.qubits
                Yterm = ["I"]*self.qubits
                Zterm = ["I"]*self.qubits

                Xterm[i] = "X"; Xterm[i+1]= "X"
                Yterm[i] = "Y"; Yterm[i+1]= "Y"
                Zterm[i] = "Z"; Zterm[i+1]= "Z"

                terms.extend( [qml.pauli.string_to_pauli_word(list_to_string(Xterm)),
                               qml.pauli.string_to_pauli_word(list_to_string(Yterm)),
                               qml.pauli.string_to_pauli_word(list_to_string(Zterm))] )
                coeff.extend( [-params["exchange"][0], -params["exchange"][1], -params["exchange"][2]] )
                
            if params["pattern"] == "open":
                pass

            else:
                Xterm = ["I"]*self.qubits
                Yterm = ["I"]*self.qubits
                Zterm = ["I"]*self.qubits

                Xterm[0] = "X"; Xterm[self.qubits-1]= "X"
                Yterm[0] = "Y"; Yterm[self.qubits-1]= "Y"
                Zterm[0] = "Z"; Zterm[self.qubits-1]= "Z"

                terms.extend( [qml.pauli.string_to_pauli_word(list_to_string(Xterm)),
                               qml.pauli.string_to_pauli_word(list_to_string(Yterm)),
                               qml.pauli.string_to_pauli_word(list_to_string(Zterm))] )
                coeff.extend( [-params["exchange"][0], -params["exchange"][1], -params["exchange"][2]] )

        elif params["pattern"] == "all_to_all":
            for i in range(self.qubits-1):
                for j in range(i+1, self.qubits):
                    Xterm = ["I"]*self.qubits
                    Yterm = ["I"]*self.qubits
                    Zterm = ["I"]*self.qubits

                    Xterm[i] = "X"; Xterm[j]= "X"
                    Yterm[i] = "Y"; Yterm[j]= "Y"
                    Zterm[i] = "Z"; Zterm[j]= "Z"

                    terms.extend( [qml.pauli.string_to_pauli_word(list_to_string(Xterm)),
                               qml.pauli.string_to_pauli_word(list_to_string(Yterm)),
                               qml.pauli.string_to_pauli_word(list_to_string(Zterm))] )
                    coeff.extend( [-params["exchange"][0], -params["exchange"][1], -params["exchange"][2]] )
        

        terms, coeff = qml.pauli.group_observables(observables=terms,coefficients=coeff, grouping_type='qwc', method='rlf')
        Pauli_terms = []
        for group in terms:
            aux = []
            for term in group:
                string = Pauli_function(term, self.qubits)
                aux.append(string)
            Pauli_terms.append(aux)

        self.hamiltonian_object = Pauli_terms
        self.coeff_object = np.hstack(coeff)
        self.parity_terms = np.array([ parity(i, self.spin, self.qubits) for i in range(2**(self.qubits*self.correction)) ]) 
        return

    def set_group_characteristics(self):
        aux_char = []
        for group in self.hamiltonian_object:
            aux_char.append( group_string(group) )
        self.groups_caractericts = aux_char
        return
    
    def cost_function(self, theta, beta):
        ansatz = theta[self.qubits:]
        dist_params = theta[:self.qubits]
        distribution = prob_dist(dist_params)
        combos = itertools.product([0, 1], repeat=self.qubits)
        s = [list(c) for c in combos]
        
        result = 0.0
        for state in s:
            expval = []
            for i,group in enumerate(self.hamiltonian_object):
                result_probs = self.node(theta = ansatz, obs = group, characteristic=self.groups_caractericts[i], state= state)
                for k,probs in enumerate(result_probs):
                    if is_identity(group[k]):
                        expval.append(1.0)
                    else:
                        expval.append( np.sum(probs*self.parity_terms[:probs.shape[0]]) )
            result_aux = np.sum( self.coeff_object*np.array(expval) )
            
            #Ponderacion termica
            for j in range(0, len(state)):
                result_aux = result_aux * distribution[j][state[j]]
            result += result_aux
        
        #Valor final
        entropy = calculate_entropy(distribution)
        final_cost = beta * result - entropy
        return final_cost