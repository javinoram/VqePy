import pennylane as qml
from pennylane import numpy as np
from quantumsim.optimizers.funciones import *

class upccgsd_ansatz():
    base = ""
    backend = ""
    token = ""

    device= None
    node = None
    node_overlap = None
    
    qubits = 0
    repetition = 0
    begin_state = None

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
    

    def set_node_overlap(self, params) -> None:
        try:
            self.repetition = params['repetitions']
        except:
            raise Exception("Number of repetitions was not indicated")

        self.node_overlap = qml.QNode(self.swap_test, qml.device(self.base, wires=2*self.qubits), interface=params['interface'])
        return
    
    
    def set_state(self, electrons):
        try:
            self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)
        except:
            raise Exception("Number of electrons should be a positive integer")


    def circuit(self, theta, obs):
        theta = theta.reshape( qml.kUpCCGSD.shape(k=self.repetition, 
            n_wires=self.qubits, delta_sz=0) )
        
        qml.kUpCCGSD(theta, wires=range(self.qubits),
            k=self.repetition, delta_sz=0, init_state=self.begin_state)
        return [qml.expval(term) for term in obs ]