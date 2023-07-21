from quantumsim.ansatz import *
from quantumsim.optimizers import *
from pennylane import qchem
from pennylane import numpy as np

class structure_molecular(upccgsd_ansatz):
    symbols = None
    coordinates = None

    groups_caractericts = None
    coeff_object = None
    parity_terms = None
    begin_state = None

    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'
    spin = 0.5

    def __init__(self, symbols, coordinates, params= None):
        self.symbols = symbols
        self.coordinates = coordinates
        
        if params['mapping']:
            if params['mapping'] in ("jordan_wigner", "bravyi_kitaev"):
                self.mapping = params['mapping']
            else:
                raise Exception("Mapping no valido, considere jordan_wigner o bravyi_kitaev")
        
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

        _, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method)
        
        self.begin_state = qml.qchem.hf_state(int(self.qubits/2), self.qubits)
        self.parity_terms = np.array([ parity(i, self.spin, self.qubits) for i in range(2**self.qubits) ]) 
        return


    def set_group_characteristics(self, hamiltonian_object):
        return np.array( [group_string(group) for group in hamiltonian_object] )

    
    def process_group(self, theta, h_object, c_object, coeff_object):
        term = h_object
        charac = c_object

        result_probs = self.node( theta=theta, obs=term, characteristic=charac )

        expval = np.array([ np.sum(probs) if is_identity(term[k]) else np.sum(probs @ self.parity_terms[:probs.shape[0]]) for k, probs in enumerate(result_probs) ])

        result = np.array( coeff_object @ expval)
        return np.sum( result )
    

    def grad_x(self, theta, x):
        grad = []
        delta = 0.01

        for i in range(len(x)):
            shift = np.zeros_like(x)
            shift[i] += 0.5 * delta

            coeff, terms = ((self.H(x + shift) - self.H(x - shift)) * delta**-1).terms()

            if len(terms)==0:
                grad.append( 0.0 )
            else:
                terms, coeff = qml.pauli.group_observables(observables=terms, coefficients=coeff, grouping_type='qwc', method='rlf')
                Pauli_terms = [] 
                for group in terms: 
                    aux = [ Pauli_function(term, self.qubits) for term in group ]
                    Pauli_terms.append(aux)
                charactericts = self.set_group_characteristics(Pauli_terms)
                
                expval = np.array( [ self.process_group(theta, Pauli_terms[i], charactericts[i], coeff[i]) for i in range(len(charactericts)) ] )
                grad.append( np.sum( expval ) )
        return np.array(grad)
    

    def H(self, x):
        return qml.qchem.molecular_hamiltonian(self.symbols, x, mult= self.mult, charge=self.charge)[0]
    

    def cost_function(self, theta, x):
        coeff, terms = self.H(x).terms()

        terms, coeff = qml.pauli.group_observables(observables=terms, coefficients=coeff, grouping_type='qwc', method='rlf')
        Pauli_terms = [] 
        for group in terms: 
            aux = [ Pauli_function(term, self.qubits) for term in group ]
            Pauli_terms.append(aux)
        charactericts = self.set_group_characteristics(Pauli_terms)
            
        result = np.array( [ self.process_group(theta, Pauli_terms[i], charactericts[i], coeff[i]) for i in range(len(charactericts)) ] )
        return np.sum( result )