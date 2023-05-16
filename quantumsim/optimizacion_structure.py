from quantumsim.functions.ansatz import *
from quantumsim.functions.constans import *
from pennylane import qchem
from pennylane import numpy as np

def finite_diff(f, x, delta=0.01):
        gradient = []

        for i in range(len(x)):
            shift = np.zeros_like(x)
            shift[i] += 0.5 * delta
            res = (f(x + shift) - f(x - shift)) * delta**-1
            gradient.append(res)

        return gradient

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

        _, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates,
            mapping= self.mapping,
            charge= 1,
            mult= self.mult,
            basis= self.basis,
            method= self.method)
        return

    def grad_x(self, theta, x):
        params = [theta[:len(self.singles)*self.repetition], theta[len(self.singles)*self.repetition:]]
        grad_h = finite_diff(self.H, x)
        grad = [self.node( params, obs ) for obs in grad_h]
        return np.array(grad)
    
    def H(self, x):
        return qml.qchem.molecular_hamiltonian(self.symbols, x, mult= self.mult, charge=self.charge)[0]
    
    
    def cost_function(self, theta, x):
        params = [theta[:len(self.singles)*self.repetition], theta[len(self.singles)*self.repetition:]]
        hamiltonian = self.H(x)
        result = self.node( params, hamiltonian )
        return result