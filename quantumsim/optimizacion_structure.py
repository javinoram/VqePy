from quantumsim.functions.ansatz import *
from quantumsim.functions.constans import *
from pennylane import qchem
from pennylane import numpy as np

class optimization_structure(given_ansatz):
    symbols = None
    coordinates = None
    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'

    def __init__(self, symbols, coordinates, params= None):
        self.symbols = symbols
        self.coordinates = coordinates

        if params['mapping']:
            if params['mapping'] in ("jordan_wigner", "bravyi_kitaev"):
                self.mapping = params['mapping']
            else:
                raise Exception("Mapping no valido, considere jordan_wigner o bravyi_kitaev")
            
        elif params['charge']:
            self.charge = params['charge']

        elif params['mult']:
            self.mult = params['mult']

        elif params['basis']:
            if params['basis'] in ("sto-3g", "6-31g", "6-311g", "cc-pvdz"):
                self.basis = params['basis']
            else:
                raise Exception("Base no valida, considere sto-3g, 6-31g, 6-311g, cc-pvdz")
        
        elif params['method']:
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
        return

    def grad_x(self, theta, x):
        params = [theta[:len(self.singles)*self.repetition], theta[len(self.singles)*self.repetition:]]
        
        grad = []
        delta = 0.01
        for i in range(len(x)):
            shift = np.zeros_like(x)
            shift[i] += 0.5 * delta
            res = (self.H(x + shift) - self.H(x - shift)) * delta**-1
            grad.append(self.node( params, res ))
        #grad_h = finite_diff(self.H, x)
        #grad = [self.node( params, obs ) for obs in grad_h]
        return np.array(grad)
    
    def H(self, x):
        return qml.qchem.molecular_hamiltonian(self.symbols, x, mult= self.mult, charge=self.charge)[0]
    
    def cost_function(self, theta, x):
        params = [theta[:len(self.singles)*self.repetition], theta[len(self.singles)*self.repetition:]]
        hamiltonian = self.H(x)
        result = self.node( params, hamiltonian )
        return result