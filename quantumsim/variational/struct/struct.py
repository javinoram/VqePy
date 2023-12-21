from quantumsim.ansatz import *
from quantumsim.optimizers import *
from pennylane import qchem
from pennylane import numpy as np

class structure_molecular():
    symbols = None
    coordinates = None
    begin_state = None

    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'

    node = None
    interface = None

    def __init__(self, symbols, coordinates, params= None):
        self.symbols = symbols
        self.coordinates = coordinates
        
        if params['mapping']:
            self.mapping = params['mapping']
        
        if params['charge']:
            self.charge = params['charge']

        if params['mult']:
            self.mult = params['mult']

        if 'basis' in params:
            self.basis = params['basis']
        
        if params['method']:
            self.method = params['method']

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
    
    def set_node(self, node, interface) -> None:
        self.node = node
        self.interface = interface
        return
    
    
    def grad_x(self, theta, x):
        delta = 0.01
        n = len(x)
        shift = np.eye( n ) * 0.5 * delta
        grad = [ self.node( theta=theta, obs=((self.H(x + shift[i]) - self.H(x - shift[i])) / delta) ) for i in range(n)]
        #shift = np.zeros_like(x)

        #for i in range(len(x)):
        #    shift[i] += 0.5 * delta

        #    hamiltonian = ((self.H(x + shift) - self.H(x - shift)) / delta)
            
        #    result = self.node( theta=theta, obs=hamiltonian )

        #    shift[i] = 0.0
        #    grad.append( result )
        return np.array(grad)
    

    def H(self, x):
        return qml.qchem.molecular_hamiltonian(self.symbols, x, mult= self.mult, charge=self.charge)[0]
    

    def cost_function(self, theta, x):
        hamiltonian = self.H(x)
        result = self.node( theta=theta, obs=hamiltonian )
        return result