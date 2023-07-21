import pennylane as qml
from pennylane import numpy as np
from quantumsim.optimizers.funciones import *


'''
Given ansatz adaptado para trabajar en el modelo molecular
'''
class given_ansatz():
    def circuit(self, theta, obs):
        pass

    base = ""
    backend = ""
    token = ""

    device= qml.device("default.qubit", wires=0)
    node = qml.QNode(circuit, device, interface="autograd")

    qubits = 0
    singles=  []
    doubles= []
    repetition = 0
    hf_state = None

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

    def set_hiperparams_circuit(self, params) -> None:
        self.repetition = params['repetitions']
        self.hf_state = qml.qchem.hf_state(params['electrons'], self.qubits)
        self.singles, self.doubles = qml.qchem.excitations(params['electrons'], self.qubits)
        return
    
    def set_node(self, params) -> None:
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return

    def circuit(self, theta, obs):
        qml.BasisState(self.hf_state, wires=range(self.qubits))
        for i in range(0, self.repetition):
            for term in self.singles:
                qml.SingleExcitation(theta[0][i], wires=term)

            for term in self.doubles:
                qml.DoubleExcitation(theta[1][i], wires=term)

        basis_change = ['I']*self.qubits
        to_measure = []
        
        for i, term in enumerate(obs):
            aux = []
            for j,string in enumerate(term[1]): 
                if string == 'X':
                    if basis_change[j] == 'I': 
                        qml.Hadamard(wires=[j])
                        basis_change[j] == 'X'
                    aux.append(j)

                elif string == 'Y':
                    if basis_change[j] == 'I': 
                        qml.S(wires=[j])
                        qml.Hadamard(wires=[j])
                        basis_change[j] == 'Y'
                    aux.append(j)

                elif string == 'Z':
                    if basis_change[j] == 'I': 
                        basis_change[j] == 'Z'
                    aux.append(j)

                else:
                    pass
            to_measure.append(aux)
        return [qml.probs(wires=to) for to in to_measure]