import pennylane as qml
from pennylane import numpy as np
from quantumsim.optimizers.funciones import *

class custom_ansatz():
    base = ""
    backend = ""
    token = ""

    device= None
    node = None
    
    qubits = 0
    begin_state = None
    excitations = []

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
        self.repetition = params['repetitions']
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return
    
    
    def set_state(self, electrons):
        self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)


    def set_gates(self, singles=None, doubles=None):
        if singles != None:
            for term in singles:
                self.excitations.append(term)
        if doubles != None:
            for term in doubles:
                self.excitations.append(term)
        return
    

    def circuit(self, theta, obs):
        qml.BasisState(self.begin_state, wires=range(self.qubits))

        for i, gate in enumerate(self.excitations):
            if len(gate) == 4:
                qml.DoubleExcitation(theta[i], wires=gate)
            elif len(gate) == 2:
                qml.SingleExcitation(theta[i], wires=gate)

        return [qml.expval(term) for term in obs ]
    