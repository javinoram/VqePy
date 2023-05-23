from quantumsim.functions.ansatz import *
from quantumsim.functions.constans import *
from pennylane import qchem
import math

'''
Clases que representan implementaciones del metodo variational quantum eigensolver (VQE).

Cada clase representa un tipo concreto de hamiltoniano en el que puede aplicar VQE,
dentro de cada uno estan los metodos necesarios para poder contruir el paso de optimizacion.
(hamiltoniano, ansatz y funcion de coste)
'''

class variational_quantum_eigensolver_electronic(given_ansatz):
    hamiltonian_object= None
    symbols = None
    coordinates = None
    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'

    '''
    Iniciador de la clase que construye el hamiltoniano molecular, todas las variables
    son guardadas en la clase
    input:
        symbols: list [string]
        coordinates: list [float]
        params: dict
    return:
        result: none
    '''
    def __init__(self, symbols, coordinates, params= None):
        self.symbols = symbols
        self.coordinates = coordinates
        self.hamiltonian_object, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method)
        return
    
    '''
    Funcion de coste: Funcion de coste que utiliza la funcion de valor esperado de qml
    input:
        theta: list [float]
    return:
        result: float
    '''
    def cost_function(self, theta):
        params = [theta[:len(self.singles)*self.repetition], theta[len(self.singles)*self.repetition:]]
        result = self.node(theta = params, obs = self.hamiltonian_object)
        return result
    

class variational_quantum_eigensolver_spin(spin_ansatz):
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
        self.spin = params['spin']
        self.correction = math.ceil( (int( 2*self.spin+1 ))/2  )
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
    def cost_function(self, theta):
        ansatz = theta
        result= 0.0
        for i, term in enumerate(self.hamiltonian_index):
            result_term = self.node( theta=ansatz, obs= term, pauli= self.hamiltonian_object[i][0])
            exchange = self.hamiltonian_object[i][1]
            result += exchange*(result_term[0] - result_term[1] - result_term[2] +result_term[3])
        return result
    
