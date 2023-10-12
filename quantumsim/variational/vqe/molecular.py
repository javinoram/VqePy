from .base import *

'''
Construtor del hamiltoniano molecular
'''
class vqe_molecular(vqe_base):
    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'
    active_electrons = None
    active_orbitals = None

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

        
        aux_h, self.qubits = qchem.molecular_hamiltonian(
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
        coeff, terms = aux_h.terms()
        del aux_h

        self.hamiltonian, self.coeff = qml.pauli.group_observables(observables=terms, coefficients=np.real(np.array(coeff)), 
                grouping_type='qwc', method='rlf')
        return

