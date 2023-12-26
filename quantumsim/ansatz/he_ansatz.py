import pennylane as qml
from pennylane import numpy as np
from .base import *

'''
Hardware Efficient ansatz
'''
class he_ansatz(base_ansatz):
    pattern = "chain"
    begin_state = None

    def set_state(self, state):
        self.begin_state = state

    '''
    Funciones para la construccion del ansatz
    '''
    def single_rotation(self, params):
        for i in range(0, self.qubits):
            qml.RY(params[i], wires=[i])
        return

    def non_local_gates(self, flag=0):
        ## Compuertas en orden normal
        if flag == 0:
            if self.pattern == 'chain':
                for i in range(0, self.qubits-1):
                    qml.CNOT(wires=[i, i+1])

            elif self.pattern == 'ring':
                if self.qubits == 2:
                    qml.CNOT(wires=[0, 1])
                else:
                    for i in range(0, self.qubits-1):
                        qml.CNOT(wires=[i, i+1])

                    qml.CNOT(wires=[self.qubits-1, 0])  

        #Compuertas en orden inverso (la operacion inversa)
        else:
            if self.pattern == 'chain':
                for i in range(self.qubits-1,0,-1):
                    qml.CNOT(wires=[i-1, i])

            elif self.pattern == 'ring':
                if self.qubits == 2:
                    qml.CNOT(wires=[0, 1])
                else:
                    qml.CNOT(wires=[self.qubits-1, 0])  
                        
                    for i in range(self.qubits-1,0,-1):
                        qml.CNOT(wires=[i-1, i])
        return
    

    def circuit(self, theta, obs):
        qml.StatePrep(self.begin_state, wires=range(self.qubits))
        rotation_number = self.qubits
        for k in range(0, self.repetition):
            params = theta[k*rotation_number:(k+1)*rotation_number]
            self.single_rotation(params)
            self.non_local_gates(0)
            
        return qml.expval(obs)


    def circuit_state(self, theta):
        qml.StatePrep(self.begin_state, wires=range(self.qubits))
        rotation_number = self.qubits
        for k in range(0, self.repetition):
            params = theta[k*rotation_number:(k+1)*rotation_number]
            self.single_rotation(params)
            self.non_local_gates(0)
        return qml.state()

