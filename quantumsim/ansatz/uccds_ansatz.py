import pennylane as qml
from pennylane import numpy as np
from quantumsim.optimizers.funciones import *

class uccds_ansatz():
    def circuit(self, theta, obs):
        pass

    base = ""
    backend = ""
    token = ""

    device= qml.device("default.qubit", wires=0)
    node = qml.QNode(circuit, device, interface="autograd")
    node_overlap = qml.QNode(circuit, device, interface="autograd")
    
    qubits = 0
    repetition = 0
    begin_state = None
    singles = None
    doubles = None

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
        if params['repetitions']:
            self.repetition = params['repetitions']
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        self.node_overlap = qml.QNode(self.swap_test, qml.device(self.base, wires=2*self.qubits), interface=params['interface'])
        return
    

    
    def set_state(self, electrons):
        self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)
        singles, doubles = qml.qchem.excitations(electrons, self.qubits)
        self.singles, self.doubles = qml.qchem.excitations_to_wires(singles, doubles)



    def circuit(self, theta, obs, characteristic):
        qml.UCCSD(theta, range(self.qubits), self.singles, self.doubles, self.begin_state)
        
        for j, index in enumerate(characteristic):
            if index == 'X':
                for k in range(self.correction):
                    qml.Hadamard(wires=[self.correction*j + k])
            elif index == 'Y':
                for k in range(self.correction):
                    qml.S(wires=[self.correction*j + k])
                    qml.Hadamard(wires=[self.correction*j + k])
            else: 
                pass
        return [qml.probs(wires=[0]) if is_identity(term) else qml.probs(wires=find_different_indices(term, "I") ) for term in obs ]
    

    def swap_test(self, theta, theta_overlap):
        #state = np.concatenate((self.begin_state, self.begin_state), axis=0)
        #qml.BasisStatePreparation(state, wires=range(2*self.qubits))

        qml.UCCSD(weights= theta, wires=range(self.qubits), 
                s_wires=self.singles, d_wires=self.doubles, init_state=self.begin_state)
        
        qml.UCCSD(weights=theta_overlap, wires=range(self.qubits),#[self.qubits+i for i in range(self.qubits)], 
                s_wires=self.singles, d_wires=self.doubles, init_state=None)

        for i in range(self.qubits):
            qml.CNOT(wires=[i,i+self.qubits])
            qml.Hadamard(wires=[i])
        
        return [qml.probs(wires=[i] ) for i in range(2*self.qubits)]