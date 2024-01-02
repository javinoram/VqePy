from .base import *

"""
Clase del modelo de moleculas, esta se encarga de construir el hamiltoniano,
hereda metodos de la clase adap_base
"""
class adap_molecular(adap_base):
    """
    Variables extras de la clase
    """
    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'
    active_electrons = None
    active_orbitals = None


    """
    Constructor de la clase
        symbols: lista con los elementos de la molecula
        coordinates: vector con las posiciones de los elementos en el espacio
        params: diccionario con los parametros del hamiltoniano
    """
    def __init__(self, symbols, coordinates, params= None):
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

        self.hamiltonian, self.qubits = qchem.molecular_hamiltonian(
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
        return
