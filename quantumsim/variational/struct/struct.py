from quantumsim.ansatz import *
from quantumsim.optimizers import *
from pennylane import qchem
from pennylane import numpy as np

"""
Clase del proceso relajacion de estructuras moleculares.
"""
class structure_molecular():
    """
    Variables de la clase
    """
    symbols = None
    coordinates = None
    begin_state = None
    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'
    active_electrons = None
    active_orbitals = None
    node = None
    interface = None


    """
    Constructor de la clase
        symbols: lista con los elementos de la molecula
        coordinates: vector con las posiciones de los elementos en el espacio
        params: diccionario con los parametros del hamiltoniano
    """
    def __init__(self, symbols, coordinates, params= None):
        self.symbols = symbols
        self.coordinates = coordinates
        
        if 'mapping' in params:
            self.mapping = params['mapping']
            
        if 'charge' in params:
            self.charge = params['charge']

        if 'mult' in params:
            self.mult = params['mult']

        if 'basis' in params:
            self.basis = params['basis']

        if 'method' in params:
            self.method = params['method']
            
        if 'active_electrons' in params:
            self.active_electrons = params['active_electrons']

        if 'active_orbitals' in params:
            self.active_orbitals = params['active_orbitals']

        _, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method,
            active_electrons=self.active_electrons,
            active_orbitals=self.active_orbitals,
            load_data=True)
        
        self.begin_state = qml.qchem.hf_state(self.active_electrons, self.qubits)
    

    """
    Funcion para setear el nodo de ejecucion del circuito
    input:
        nodo: nodo de ejecucion
        interface: interface en la cual se construyo el nodo
    """
    def set_node(self, node, interface) -> None:
        self.node = node
        self.interface = interface
    

    """
    Funcion que calcula el gradiente de la funcion de coste respecto a las posiciones de los elementos
    input:
        nodo: nodo de ejecucion
        interface: interface en la cual se construyo el nodo
    output:
        grad: retorna el gradiente
    """
    def grad_x(self, theta, x):
        delta = 0.01
        n = len(x)
        shift = np.eye( n ) * 0.5 * delta
        grad = [ self.node( theta=theta, obs=((self.H(x + shift[i]) - self.H(x - shift[i])) / delta) ) for i in range(n)]
        return np.array(grad)
    

    """
    Funcion que calcula el hamiltoniano molecular considerando las caracteristicas antes ingresadas.
    input:
        x: vector con las posiciones de los elementos
    output:
        h: hamiltoniano molecular
    """
    def H(self, x):
        h, q = qchem.molecular_hamiltonian(symbols= self.symbols,
            coordinates= x,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method,
            active_electrons=self.active_electrons,
            active_orbitals=self.active_orbitals,
            load_data=True)
        return h   


    """
    Funcion de coste.
    input:
        theta: vector de parametros del circuito
        x: vector con las posiciones de los elementos
    output:
        result: evaluacion de la funcion de coste dado theta y x.
    """
    def cost_function(self, theta, x):
        hamiltonian = self.H(x)
        result = self.node( theta=theta, obs=hamiltonian )
        return result