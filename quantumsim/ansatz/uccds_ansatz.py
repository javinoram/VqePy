import pennylane as qml
from pennylane import numpy as np
from quantumsim.optimizers.funciones import *

class uccds_ansatz():
    base = ""
    backend = ""
    token = ""

    device= None
    node = None
    qubits = 0
    repetition = 0
    correction = 1

    begin_state = None
    singles = None
    doubles = None

    def set_device(self, params) -> None:
        self.base = params['base']
        self.qubits = params["qubits"]

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
        try:
            self.repetition = params['repetitions']
        except:
            raise Exception("Number of repetitions was not indicated")
        
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return
    
    
    def set_state(self, electrons):
        try:   
            self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)
            singles, doubles = qml.qchem.excitations(electrons, self.qubits)
            self.singles, self.doubles = qml.qchem.excitations_to_wires(singles, doubles)
        except:
            raise Exception("Number of electrons should be a positive integer")



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
