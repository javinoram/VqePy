import pennylane as qml
from pennylane import numpy as np
from quantumsim.optimizers.funciones import *
from quantumsim.ansatz import *
from quantumsim.lattice import *
from pennylane import qchem
from pennylane import FermiC, FermiA

class adap_molecular():
    hamiltonian = None
    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'
    active_electrons = None
    active_orbitals = None

    base = ""
    backend = ""
    token = ""

    device= None
    node = None
    
    qubits = 0
    begin_state = None

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

        
        self.hamiltonian, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method,
            active_electrons=self.active_electrons, 
            active_orbitals=self.active_orbitals)
        return

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


    def set_hamiltonian(self, hamiltonian):
        self.hamiltonian = hamiltonian


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
    


class adap_fermihubbard():
    hamiltonian = None
    hopping = 0.0
    potential = 0.0
    electric = 0.0

    base = ""
    backend = ""
    token = ""

    device= None
    node = None
    
    qubits = 0
    begin_state = None

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

        
        self.hamiltonian, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method,
            active_electrons=self.active_electrons, 
            active_orbitals=self.active_orbitals)
        return

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

    def set_hamiltonian(self, hamiltonian):
        self.hamiltonian = hamiltonian

    def circuit(self):
        [qml.PauliX(i) for i in np.nonzero(self.begin_state)[0]]
        return qml.expval(self.hamiltonian)
    