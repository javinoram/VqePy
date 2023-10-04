from quantumsim.ansatz import *
from quantumsim.lattice import *
from pennylane import qchem
from pennylane import FermiC, FermiA
import math

'''
Clase con las funciones de coste para utilizar VQE y VQD en 
un hamiltoniano molecular
'''
class vqe_molecular():
    hamiltonian_object= None
    coeff_object = None
    qubits = 0

    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'
    active_electrons = None
    active_orbitals = None

    node = None
    node_overlap = None

    def __init__(self, symbols, coordinates, params= None):
        if 'mapping' in params:
            if params['mapping'] in ("jordan_wigner"):
                self.mapping = params['mapping']
            else:
                raise Exception("Mapping no valido, considere jordan_wigner")
            
        if 'charge' in params:
            self.charge = params['charge']

        if 'mult' in params:
            self.mult = params['mult']

        if 'basis' in params:
            if params['basis'] in ("sto-3g", "6-31g", "6-311g", "cc-pvdz"):
                self.basis = params['basis']
            else:
                raise Exception("Base no valida, considere sto-3g, 6-31g, 6-311g, cc-pvdz")
        
        if 'method' in params:
            if params['method'] in ("pyscf", "dhf"):
                self.method = params['method']
            else:
                raise Exception("Metodo no valido, considere dhf o pyscf")
            
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
            active_orbitals=self.active_orbitals)
        coeff, terms = aux_h.terms()
        del aux_h

        self.hamiltonian_object, self.coeff_object = qml.pauli.group_observables(observables=terms, coefficients=coeff, 
                grouping_type='qwc', method='rlf')
        return
    

    def process_group(self, theta, i):
        result = self.node( theta=theta, obs= self.hamiltonian_object[i] )
        result = np.array(self.coeff_object[i]) @ np.array(result)
        return np.sum( result )
    

    def cost_function(self, theta):
        results = [ self.process_group(theta, i) for i in range(len(self.hamiltonian_object)) ]
        return np.sum( results )



'''
Clase con las funciones de coste para utilizar VQE y VQD
en un hamiltoniano de espines
'''
class vqe_spin():
    hamiltonian_object = None
    coeff_object = None

    node = None
    node_overlap = None

    def __init__(self, params, lat):
        self.qubits = params['sites']
        self.spin = params['spin']

        terms = []
        coeff = []

        if lat['lattice'] == 'custom':
            pass
        else:
            x,y = lat['size']
            lattice_edge, lattice_node = lattice(lat)

            #J term
            for pair in lattice_edge:
                termz = qml.PauliZ(wires=[ x*pair[0][0] + pair[0][1], x*pair[1][0] + pair[1][1]])
                termx = qml.PauliX(wires=[ x*pair[0][0] + pair[0][1], x*pair[1][0] + pair[1][1]])
                termy = qml.PauliY(wires=[ x*pair[0][0] + pair[0][1], x*pair[1][0] + pair[1][1]])

                terms.extend( [termx, termy, termz] )
                coeff.extend( [-params["J"][0]/4.0, -params["J"][1]/4.0, -params["J"][2]/4.0] )

            #h term
            if 'magnetic' in params['interactions']:
                for pair in lattice_node:
                    termz = qml.PauliZ(wires=[ x*pair[0] + pair[1]])
                    termz = qml.PauliX(wires=[ x*pair[0] + pair[1]])
                    termz = qml.PauliY(wires=[ x*pair[0] + pair[1]])

                    terms.extend( [termx, termy, termz] )
                    coeff.extend( [params["h"][0]/2.0, params["h"][1]/2.0, params["h"][2]/2.0] )
        
        to_delete = []
        for i,c in enumerate(coeff):
            if c==0.0:
                to_delete.append(i)
        
        to_delete = -np.sort(-np.array(to_delete))
        for index in to_delete:
            coeff.pop(index)
            terms.pop(index)

        self.hamiltonian_object, self.coeff_object = qml.pauli.group_observables(observables=terms, coefficients=coeff, 
                grouping_type='qwc', method='rlf')
        return

    def process_group(self, theta, i):
        result = self.node( theta=theta, obs= self.hamiltonian_object[i] )
        result = np.real( np.array(self.coeff_object[i]) ) @ np.array(result)
        return np.sum( result )
    
    def cost_function(self, theta):
        results = [ self.process_group(theta, i) for i in range(len(self.hamiltonian_object)) ]
        return np.sum( results )

    def overlap_cost_function(self, theta, theta_overlap):
        result_probs = self.node_overlap(theta = theta, theta_overlap=theta_overlap)
        return result_probs[0]
    

'''
Clase con las funciones de coste para utilizar VQE y VQD
en un hamiltoniano de espines
'''
class vqe_fermihubbard():
    hamiltonian_object= None
    hopping = 0.0
    potential = 0.0
    electric = 0.0

    qubits = 0

    node = None
    node_overlap = None

    def __init__(self, params, lat):
        self.qubits = params["sites"]*2
        self.hopping = params["hopping"]
        self.potential = params["potential"]
        fermi_sentence = 0.0
        fermi_potential = 0.0

        if lat['lattice'] == 'custom':
            pass
        else:
            x,y = lat['size']
            lattice_edge, lattice_node = lattice(lat)

            #-t term
            fermi_hopping = 0.0
            for pair in lattice_edge:
                p1, p2 = pair
                fermi_hopping +=  FermiC(2*(x*p1[0] + p1[1]))*FermiA(2*(x*p2[0] + p2[1])) + FermiC(2*(x*p2[0] + p2[1]))*FermiA(2*(x*p1[0] + p1[1]))
                fermi_hopping +=  FermiC(2*(x*p1[0] + p1[1])+1)*FermiA(2*(x*p2[0] + p2[1])+1) + FermiC(2*(x*p2[0] + p2[1])+1)*FermiA(2*(x*p1[0] + p1[1])+1)
            
            #U term
            if 'on-site' in params['interactions']:
                fermi_U = 0.0
                for node in lattice_node:
                    p1, p2 = node
                    fermi_U += FermiC(2*(x*p1[0] + p1[1]))*FermiA(2*(x*p1[0] + p1[1]))*FermiC(2*(x*p2[0] + p2[1])+1)*FermiA(2*(x*p2[0] + p2[1])+1)

            #E term
            if 'electric' in params['interactions']:
                fermi_E = 0.0
                for node in lattice_node:
                    p1, p2 = node
                    fermi_U += FermiC(2*(x*p1[0] + p1[1]))*FermiA(2*(x*p1[0] + p1[1]))
                    fermi_U += FermiC(2*(x*p2[0] + p2[1])+1)*FermiA(2*(x*p2[0] + p2[1])+1)
                pass
            


        fermi_sentence = -fermi_hopping + (self.potential/self.hopping)*fermi_potential
        coeff, terms = qml.jordan_wigner( fermi_sentence, ps=True ).hamiltonian().terms()

        to_delete = []
        for i,c in enumerate(coeff):
            if c==0.0:
                to_delete.append(i)
        
        to_delete = -np.sort(-np.array(to_delete))
        for index in to_delete:
            coeff.pop(index)
            terms.pop(index)

        self.hamiltonian_object, self.coeff_object = qml.pauli.group_observables(observables=terms, coefficients=coeff, 
                grouping_type='qwc', method='rlf')
        return

    def process_group(self, theta, i):
        result = self.node( theta=theta, obs= self.hamiltonian_object[i] )
        result = np.real( np.array(self.coeff_object[i]) ) @ np.array(result)
        return np.sum( result )
    
    def cost_function(self, theta):
        results = [ self.process_group(theta, i) for i in range(len(self.hamiltonian_object)) ]
        return np.sum( results )