from quantumsim.ansatz import *
from quantumsim.optimizers import *
from pennylane import qchem
from pennylane import numpy as np

class structure_molecular():
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

    node = None

    def __init__(self, symbols, coordinates, params= None):
        self.symbols = symbols
        self.coordinates = coordinates
        
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
            if params['basis'] in ("sto-3g"):
                self.basis = params['basis']
            else:
                raise Exception("Base no valida, considere sto-3g")
        
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
        return
    
    
    def process_group(self, theta, h_object, coeff_object):
        expval = np.array( self.node( theta=theta, obs=h_object) )
        coeff = np.array(coeff_object)
        result = np.array( coeff @ expval)
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
                expval = np.array( [ self.process_group(theta, terms[i], coeff[i]) for i in range(len(terms)) ] )
                grad.append( np.sum( expval ) )
        return np.array(grad)
    

    def H(self, x):
        return qml.qchem.molecular_hamiltonian(self.symbols, x, mult= self.mult, charge=self.charge)[0]
    

    def cost_function(self, theta, x):
        coeff, terms = self.H(x).terms()
        terms, coeff = qml.pauli.group_observables(observables=terms, coefficients=coeff, grouping_type='qwc', method='rlf')
        result = np.array( [ self.process_group(theta, terms[i], coeff[i]) for i in range(len(terms)) ] )
        result = np.sum( result )
        return result 