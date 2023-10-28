from quantumsim.ansatz import *
from quantumsim.lattice import *
from pennylane import qchem
from pennylane import FermiC, FermiA
import itertools


'''
Base del VQE donde se definen las funciones de coste,
y funciones de calculos auxiliares, como las proyecciones
'''
class vqe_base():
    hamiltonian= None
    coeff = None
    qubits = 0
    node = None

    def process_group(self, theta, i):
        result = self.node( theta=theta, obs= self.hamiltonian[i] )
        result = np.array(self.coeff[i]) @ np.array(result)
        return result 
    
    def cost_function(self, theta):
        results = [ self.process_group(theta, i) for i in range(len(self.hamiltonian)) ]
        return np.sum( results )
    
    #Cantidades utiles de saber
    #Probabilidad de cada uno de los estados basales segun el estado obtenido
    def get_projections(self, theta):
        combos = itertools.product([0, 1], repeat=self.qubits)
        s = [list(c) for c in combos]
        pro = [ qml.Projector(state, wires=range(self.qubits)) for state in s]
        result = [self.node( theta=theta, obs=[state])[0] for state in pro]
        return result, s
    
    #Espin total del estado
    def get_totalspinS(self, theta, electrons):
        s_square = qml.qchem.spin2(electrons, self.qubits)
        result = self.node( theta=theta, obs=[s_square] )[0]
        return result
    
    #Espin total proyectado en S_z del estado
    def get_totalspinSz(self, theta, electrons):
        s_z = qml.qchem.spinz(electrons, self.qubits)
        result = self.node( theta=theta, obs=[s_z] )[0]
        return result
    
    #Numero de particulas del estado
    def get_totalspinSz(self, theta, electrons):
        n = qml.qchem.particle_number(self.qubits)
        result = self.node( theta=theta, obs=[n] )[0]
        return result


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



'''
Construtor del hamiltoniano de espines
'''
class vqe_spin(vqe_base):
    def __init__(self, params, lat):
        self.qubits = params['sites']
        terms = []
        coeff = []

        if lat['lattice'] == 'custom':
            pass
        else:
            x,y = lat['size']
            lattice_edge, lattice_node = lattice(lat)

            #J term
            for pair in lattice_edge:
                termz = qml.PauliZ(wires=[ x*pair[0][0] + pair[0][1]])@qml.PauliZ(wires=[x*pair[1][0] + pair[1][1]])
                termx = qml.PauliX(wires=[ x*pair[0][0] + pair[0][1]])@qml.PauliX(wires=[x*pair[1][0] + pair[1][1]])
                termy = qml.PauliY(wires=[ x*pair[0][0] + pair[0][1]])@qml.PauliY(wires=[x*pair[1][0] + pair[1][1]])

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

        self.hamiltonian, self.coeff = qml.pauli.group_observables(observables=terms, coefficients=np.real(np.array(coeff)), 
                grouping_type='qwc', method='rlf')
        return
    

'''
Construtor del hamiltoniano de fermi-hubbard
'''
class vqe_fermihubbard(vqe_base):
    def __init__(self, params, lat):
        self.qubits = params["sites"]*2
        fermi_sentence = 0.0

        if lat['lattice'] == 'custom':
            pass
        else:
            x,y = lat['size']
            lattice_edge, lattice_node = lattice(lat)

            #-t term
            hopping = -params["hopping"]
            fermi_hopping = 0.0
            for pair in lattice_edge:
                p1, p2 = pair
                fermi_hopping +=  FermiC(2*(x*p1[0] + p1[1]))*FermiA(2*(x*p2[0] + p2[1])) + FermiC(2*(x*p2[0] + p2[1]))*FermiA(2*(x*p1[0] + p1[1]))
                fermi_hopping +=  FermiC(2*(x*p1[0] + p1[1])+1)*FermiA(2*(x*p2[0] + p2[1])+1) + FermiC(2*(x*p2[0] + p2[1])+1)*FermiA(2*(x*p1[0] + p1[1])+1)

            fermi_sentence = hopping*fermi_hopping

            #U term
            if 'on-site' in params['interactions']:
                Upotential = params["potential"]
                fermi_U = 0.0
                for node in lattice_node:
                    p1, p2 = node
                    fermi_U += FermiC(x*p1 + 2*p2)*FermiA(x*p1 + 2*p2)*FermiC(x*p1 + 2*p2+1)*FermiA(x*p1 + 2*p2+1)
                fermi_sentence += Upotential*fermi_U

            #E term
            if 'electric' in params['interactions']:
                Efield = params["electric"]
                fermi_E = 0.0
                for node in lattice_node:
                    p1, p2 = node
                    fermi_E += FermiC(x*p1 + 2*p2)*FermiA(x*p1 + 2*p2)
                    fermi_E += FermiC(x*p1 + 2*p2+1)*FermiA(x*p1 + 2*p2+1)
                fermi_sentence += Efield*fermi_E


        coeff, terms = qml.jordan_wigner( fermi_sentence, ps=True ).hamiltonian().terms()

        to_delete = []
        for i,c in enumerate(coeff):
            if c==0.0:
                to_delete.append(i)
            if isinstance(terms[i], qml.Identity):
                terms[i] = qml.Identity(wires=[0])
        
        to_delete = -np.sort(-np.array(to_delete))
        for index in to_delete:
            coeff.pop(index)
            terms.pop(index)

        self.hamiltonian, self.coeff = qml.pauli.group_observables(observables=terms, coefficients=np.real(np.array(coeff)), 
                grouping_type='qwc', method='rlf')
        return