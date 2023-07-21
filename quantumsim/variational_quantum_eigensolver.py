from quantumsim.functions.ansatz import *
from quantumsim.functions.funciones import *
from pennylane import qchem
from pennylane import FermiC, FermiA
import math

'''
Clase con las funciones de coste para utilizar VQE y VQD en 
un hamiltoniano molecular
'''
class vqe_molecular(HE_ansatz):
    hamiltonian_object= None
    groups_caractericts = None
    coeff_object = None
    parity_terms = None

    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'

    def __init__(self, symbols, coordinates, params= None):
        if params['mapping']:
            if params['mapping'] in ("jordan_wigner"):
                self.mapping = params['mapping']
            else:
                raise Exception("Mapping no valido, considere jordan_wigner")
            
        if params['charge']:
            self.charge = params['charge']

        if params['mult']:
            self.mult = params['mult']

        if params['basis']:
            if params['basis'] in ("sto-3g", "6-31g", "6-311g", "cc-pvdz"):
                self.basis = params['basis']
            else:
                raise Exception("Base no valida, considere sto-3g, 6-31g, 6-311g, cc-pvdz")
        
        if params['method']:
            if params['method'] in ("pyscf", "dhf"):
                self.method = params['method']
            else:
                raise Exception("Metodo no valido, considere dhf o pyscf")

        
        aux_h, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method)
        coeff, terms = aux_h.terms()
        del aux_h

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
        self.parity_terms = np.array([ parity(i) for i in range(2**self.qubits) ]) 
        return
    

    def set_group_characteristics(self):
        aux_char = []
        for group in self.hamiltonian_object:
            aux_char.append( group_string(group) )
        self.groups_caractericts = aux_char
        return
    

    def cost_function(self, theta, state):
        expval = []
        for i,group in enumerate(self.hamiltonian_object):
            result_probs = self.node(theta = theta, obs = group, characteristic=self.groups_caractericts[i], state= state)
            for k,probs in enumerate(result_probs):
                if is_identity(group[k]):
                    expval.append(1.0)
                else:
                    expval.append( np.sum(probs*self.parity_terms[:probs.shape[0]]) )
        return np.sum( self.coeff_object*np.array(expval) )
    

    def overlap_cost_function(self, theta, theta_overlap, state, state_overlap):
        result_probs = self.node_overlap(theta = theta, theta_overlap=theta_overlap, state= state, state_overlap=state_overlap)
        return result_probs[0]



'''
Clase con las funciones de coste para utilizar VQE y VQD
en un hamiltoniano de espines
'''
class vqe_spin(HE_ansatz):
    hamiltonian_object = None
    groups_caractericts = None
    coeff_object = None
    parity_terms = None

    def __init__(self, params):
        self.qubits = params['sites']
        self.spin = params['spin']
        self.correction = math.ceil( (int( 2*self.spin+1 ))/2  )

        terms = []
        coeff = []
        self.hamiltonian_object = None

        if params["pattern"] == "open":
            for i in range(self.qubits-1):
                Xterm = ["I" for _ in range(self.qubits)]
                Yterm = ["I" for _ in range(self.qubits)]
                Zterm = ["I" for _ in range(self.qubits)]

                Xterm[i] = "X"; Xterm[i+1]= "X"
                Yterm[i] = "Y"; Yterm[i+1]= "Y"
                Zterm[i] = "Z"; Zterm[i+1]= "Z"

                terms.append( qml.pauli.string_to_pauli_word(list_to_string(Xterm)) )
                terms.append( qml.pauli.string_to_pauli_word(list_to_string(Yterm)) )
                terms.append( qml.pauli.string_to_pauli_word(list_to_string(Zterm)) )

                coeff.append( -params["exchange"][0] )
                coeff.append( -params["exchange"][1] )
                coeff.append( -params["exchange"][2] )
        
        elif params["pattern"] == "close":
            for i in range(self.qubits-1):
                Xterm = ["I" for _ in range(self.qubits)]
                Yterm = ["I" for _ in range(self.qubits)]
                Zterm = ["I" for _ in range(self.qubits)]

                Xterm[i] = "X"; Xterm[i+1]= "X"
                Yterm[i] = "Y"; Yterm[i+1]= "Y"
                Zterm[i] = "Z"; Zterm[i+1]= "Z"

                terms.append( qml.pauli.string_to_pauli_word(list_to_string(Xterm)) )
                terms.append( qml.pauli.string_to_pauli_word(list_to_string(Yterm)) )
                terms.append( qml.pauli.string_to_pauli_word(list_to_string(Zterm)) )

                coeff.append( -params["exchange"][0] )
                coeff.append( -params["exchange"][1] )
                coeff.append( -params["exchange"][2] )

            Xterm = ["I" for _ in range(self.qubits)]
            Yterm = ["I" for _ in range(self.qubits)]
            Zterm = ["I" for _ in range(self.qubits)]

            Xterm[0] = "X"; Xterm[self.qubits-1]= "X"
            Yterm[0] = "Y"; Yterm[self.qubits-1]= "Y"
            Zterm[0] = "Z"; Zterm[self.qubits-1]= "Z"

            terms.append( qml.pauli.string_to_pauli_word(list_to_string(Xterm)) )
            terms.append( qml.pauli.string_to_pauli_word(list_to_string(Yterm)) )
            terms.append( qml.pauli.string_to_pauli_word(list_to_string(Zterm)) )

            coeff.append( -params["exchange"][0] )
            coeff.append( -params["exchange"][1] )
            coeff.append( -params["exchange"][2] )

        elif params["pattern"] == "all_to_all":
            pass
        

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
        self.parity_terms = np.array([ parity(i) for i in range(2**self.qubits) ]) 
        return

    def set_group_characteristics(self):
        aux_char = []
        for group in self.hamiltonian_object:
            aux_char.append( group_string(group) )
        self.groups_caractericts = aux_char
        return
    
    def cost_function(self, theta, state):
        expval = []
        for i,group in enumerate(self.hamiltonian_object):
            result_probs = self.node(theta = theta, obs = group, characteristic=self.groups_caractericts[i], state= state)
            for k,probs in enumerate(result_probs):
                if is_identity(group[k]):
                    expval.append(1.0)
                else:
                    expval.append( np.sum(probs*self.parity_terms[:probs.shape[0]]) )
        return np.sum( self.coeff_object*np.array(expval) )
    
    def overlap_cost_function(self, theta, theta_overlap, state, state_overlap):
        result_probs = self.node_overlap(theta = theta, theta_overlap=theta_overlap, state= state, state_overlap=state_overlap)
        return result_probs[0]
    

'''
Clase con las funciones de coste para utilizar VQE y VQD
en un hamiltoniano de espines
'''
class vqe_fermihubbard(HE_ansatz):
    hamiltonian_object= None
    hopping = 0.0
    potential = 0.0
    
    def __init__(self, params):
        self.qubits = params["sites"]*2
        self.hopping = params["hopping"]
        self.potential = params["potential"]

        fermi_sentence = 0.0
        coeff = []
        expression = []

        for i in range(params["sites"]-1):
            fermi_sentence +=  -self.hopping*FermiC(2*i)*FermiA(2*i +2) + -self.hopping*FermiC(2*i +2)*FermiA(2*i)
            fermi_sentence +=  -self.hopping*FermiC(2*i+1)*FermiA(2*i +3) + -self.hopping*FermiC(2*i +3)*FermiA(2*i +1)  
            fermi_sentence +=  -self.potential*FermiC(2*i)*FermiA(2*i)*FermiC(2*i +1)*FermiA(2*i +1) 

        fermi_sentence = qml.jordan_wigner( fermi_sentence )
        for term in fermi_sentence:
            coeff_aux, operator_aux = term.terms()
            if np.real(coeff_aux[0]) != 0.0:
                coeff.append( np.real(coeff_aux[0]) )
                expression.append( operator_aux[0] )

        Pauli_terms = []
        for k, term in enumerate(expression):
            if type(term) not in [qml.ops.identity.Identity, qml.ops.qubit.non_parametric_ops.PauliZ,  qml.ops.qubit.non_parametric_ops.PauliX, qml.ops.qubit.non_parametric_ops.PauliY]:
                _, aux2 = term.terms()
                decomp_list = aux2[0].decomposition()
            string = Pauli_function(decomp_list, 6)
            Pauli_terms.append( [coeff[k], string] )

        self.hamiltonian_object = conmute_group(Pauli_terms)
        return
    
    
    def cost_function(self, theta):
        params = [theta[:self.repetition], theta[self.repetition:]]
        result = 0.0

        for group in self.hamiltonian_object:
            if is_identity(group[0][1]):
                exchange = np.array( group[0][0], requires_grad=True) 
                exchange.requires_grad = True
                result += exchange
            else:
                result_probs = self.node(theta = params, obs = group)
                for k, probs in enumerate(result_probs):
                    exchange = np.array( group[k][0], requires_grad=True) 
                    exchange.requires_grad = True
                    for j in range(len(probs)):
                        result += exchange*probs[j]*parity(j)
        return np.real(result)