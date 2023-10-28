import pennylane as qml
from pennylane import numpy as np
from quantumsim.ansatz import *
from quantumsim.lattice import *
from pennylane import qchem
from pennylane import FermiC, FermiA

class adap_base():
    hamiltonian = None
    qubits = 0

    base = ""
    backend = ""
    token = ""
    device= None
    node = None
    begin_state = None

    def set_device(self, params) -> None:
        self.base = params['base']

        ##Maquinas reales
        if self.base == 'qiskit.ibmq':
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            if params['token']:
                self.token = params['token']
                self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits,  ibmqx_token= self.token)
            else:
                raise Exception("Token de acceso no encontrado")
        ##Simuladores de qiskit
        elif self.base == "qiskit.aer":
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            
            self.device= qml.device(self.base, backend=self.backend, wires=self.qubits)
        ##Simuladores de pennylane
        else:
            self.device= qml.device(self.base, wires=self.qubits)
        return

    def set_node(self, params) -> None:
        self.repetition = params['repetitions']
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return
    
    def set_state(self, electrons):
        self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)

    def circuit(self, params, excitations, params_select=None, gates_select=None):
        qml.BasisState(self.begin_state, wires=range(self.qubits))

        if gates_select != None:
            for i, gate in enumerate(gates_select):
                if len(gate) == 4:
                    qml.DoubleExcitation(params_select[i], wires=gate)
                elif len(gate) == 2:
                    qml.SingleExcitation(params_select[i], wires=gate)

        for i, gate in enumerate(excitations):
            if len(gate) == 4:
                qml.DoubleExcitation(params[i], wires=gate)
            elif len(gate) == 2:
                qml.SingleExcitation(params[i], wires=gate)
        return qml.expval(self.hamiltonian)




class adap_molecular(adap_base):
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

class adap_fermihubbard(adap_base):

    def __init__(self, params, lat):
        self.qubits = params["sites"]*2
        hopping = -params["hopping"]
        potential = params["potential"]
        fermi_sentence = 0.0

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
            fermi_sentence = hopping*fermi_hopping

            #U term
            if 'on-site' in params['interactions']:
                fermi_U = 0.0
                for node in lattice_node:
                    p1, p2 = node
                    fermi_U += FermiC(x*p1 + 2*p2)*FermiA(x*p1 + 2*p2)*FermiC(x*p1 + 2*p2+1)*FermiA(x*p1 + 2*p2+1)
                fermi_sentence += potential*fermi_U

            #E term
            if 'electric' in params['interactions']:
                fermi_E = 0.0
                for node in lattice_node:
                    p1, p2 = node
                    fermi_E += FermiC(2*(x*p1[0] + p1[1]))*FermiA(2*(x*p1[0] + p1[1]))
                    fermi_E += FermiC(2*(x*p2[0] + p2[1])+1)*FermiA(2*(x*p2[0] + p2[1])+1)
                pass#fermi_sentence += self.potential*fermi_E
        
        #fermi_sentence = self.hopping*fermi_hopping + self.potential*fermi_U
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
        self.hamiltonian = qml.Hamiltonian(np.real(np.array(coeff)), terms)
        return