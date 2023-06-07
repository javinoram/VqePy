from physics.functions.ansatz import *
from pennylane import qchem
import math

class electronic_hamiltonian(given_ansatz):
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

        self.hamiltonian_object, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates/2,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method)
        self.hamiltonian_object = self.hamiltonian_object
        return
    
    def density_charge(self, theta, time):
        number_pairs = int( self.qubits/2 )
        params = [theta[:self.repetition], theta[self.repetition:]]
        value_per_sites = []
        for i in range(number_pairs):
            result_down = self.node(theta = params, hamiltonian=self.hamiltonian_object, obs =[2*i], time=time)
            result_up = self.node(theta = params, hamiltonian=self.hamiltonian_object, obs =[2*i+1], time=time)
            value_per_sites.append(result_up[1] + result_down[1])
        return value_per_sites
    

    def density_spin(self, theta, time):
        number_pairs = int( self.qubits/2 )
        params = [theta[:self.repetition], theta[self.repetition:]]
        value_per_sites = []
        for i in range(number_pairs):
            result_down = self.node(theta = params, hamiltonian=self.hamiltonian_object, obs =[2*i], time=time)
            result_up = self.node(theta = params, hamiltonian=self.hamiltonian_object, obs =[2*i+1], time=time)
            value_per_sites.append(result_up[1] - result_down[1])
        return value_per_sites