import pennylane as qml
from pennylane import numpy as np
from quantumsim.optimizers.funciones import *

'''
Hardware Efficient ansatz
'''
class HE_ansatz():
    def circuit(self, theta, obs):
        pass

    base = ""
    backend = ""
    token = ""

    device= qml.device("default.qubit", wires=0)
    node = qml.QNode(circuit, device, interface="autograd")
    node_overlap = qml.QNode(circuit, device, interface="autograd")
    pattern = "chain"
    qubits = 0
    correction = 1
    repetition = 0

    def set_device(self, params) -> None:
        self.base = params['base']

        ## Maquinas reales
        if self.base == 'qiskit.ibmq':
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            if params['token']:
                self.token = params['token']
                self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits*self.correction, ibmqx_token= self.token)
            else:
                raise Exception("Token de acceso no encontrado")
        ## Simuladores de qiskit
        elif self.base == "qiskit.aer":
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits*self.correction)
        ## Simuladores de pennylane
        else:
            self.device= qml.device(self.base, wires=self.qubits*self.correction)
        return
    
    def set_node(self, params) -> None:
        if params["pattern"]:
            self.pattern = params["pattern"]
        
        if params['repetitions']:
            self.repetition = params['repetitions']
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        self.node_overlap = qml.QNode(self.circuit_overlap, self.device, interface=params['interface'])
        return
    

    '''
    Funciones para la construccion del ansatz
    '''
    def single_rotation(self, params, ):
        for i in range(0, self.qubits):
            for j in range(self.correction):
                qml.RY(params[i], wires=[self.correction*i+j])
        return

    def non_local_gates(self, flag=0):
        ## Compuertas en orden normal
        if flag == 0:
            if self.pattern == 'chain':
                for i in range(0, self.qubits-1):
                    for j in range(self.correction):
                        qml.CNOT(wires=[self.correction*i+j, self.correction*(i+1)+j])

            elif self.pattern == 'ring':
                if self.qubits == 2:
                    for j in range(self.correction):
                        qml.CNOT(wires=[j, self.correction+j])
                else:
                    for i in range(0, self.qubits-1):
                        for j in range(self.correction):
                            qml.CNOT(wires=[self.correction*i+j, self.correction*(i+1)+j])

                    for j in range(self.correction):
                        qml.CNOT(wires=[self.correction*(self.qubits-1)+j, j])  
            
            elif self.pattern == 'all_to_all':
                if self.qubits == 2:
                    for j in range(self.correction):
                        qml.CNOT(wires=[j, self.correction+j])
                else:
                    for i in range(0, self.qubits-1):
                        for k in range(i+1, self.qubits):
                            for j in range(self.correction):
                                qml.CNOT(wires=[self.correction*i+j, self.correction*k +j])


        #Compuertas en orden inverso (la operacion inversa)
        else:
            if self.pattern == 'chain':
                for i in range(self.qubits-1,0,-1):
                    for j in range(self.correction-1, -1, -1):
                        qml.CNOT(wires=[self.correction*(i-1) +j, self.correction*i +j])

            elif self.pattern == 'ring':
                if self.qubits == 2:
                    for j in range(self.correction-1, -1, -1):
                        qml.CNOT(wires=[j, self.correction+j])
                else:
                    for j in range(self.correction-1, -1, -1):
                        qml.CNOT(wires=[self.correction*(self.qubits-1)+j, j])  
                        
                    for i in range(self.qubits-1,0,-1):
                        for j in range(self.correction-1, -1, -1):
                            qml.CNOT(wires=[self.correction*(i-1) +j, self.correction*i +j])

            elif self.pattern == 'all_to_all':
                if self.qubits == 2:
                    for j in range(self.correction-1, -1, -1):
                        qml.CNOT(wires=[j, self.correction+j])
                else:
                    for i in range(self.qubits-1,0,-1):
                        for k in range(i-1, -1, -1):
                            for j in range(self.correction-1, -1, -1):
                                qml.CNOT(wires=[self.correction*k +j, self.correction*i+j])
        return
    

    
    def circuit(self, theta, obs, characteristic, state):
        qml.BasisState(state, wires=range(self.qubits*self.correction))
        rotation_number = self.qubits
        for k in range(0, self.repetition):
            params = theta[k*rotation_number:(k+1)*rotation_number]
            self.single_rotation(params)
            self.non_local_gates(0)
            
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


    def circuit_overlap(self, theta, theta_overlap, state, state_overlap):
        qml.BasisState(state, wires=range(self.qubits*self.correction))
        rotation_number = self.qubits
        for k in range(0, self.repetition):
            params = theta[k*rotation_number:(k+1)*rotation_number]
            self.single_rotation(params)
            self.non_local_gates(0)
        
        theta_overlap = theta_overlap[::-1]
        for k in range(0, self.repetition):
            params = -np.array(theta_overlap[k*rotation_number:(k+1)*rotation_number])[::-1]
            self.non_local_gates(1)
            self.single_rotation(params)
        
        for k in range(len(state_overlap)):
            if state_overlap[k]==1:
                qml.X(k)
            else:
                pass    
        return qml.probs(wires=[i for i in range(self.qubits)])


'''
Hardware Efficient ansatz
'''
class HE_ansatz_thermal():
    def circuit(self, theta, obs):
        pass

    base = ""
    backend = ""
    token = ""

    device= qml.device("default.qubit", wires=0)
    node = qml.QNode(circuit, device, interface="autograd")
    pattern = "chain"
    qubits = 0
    correction = 1
    repetition = 0

    def set_device(self, params) -> None:
        self.base = params['base']

        ## Maquinas reales
        if self.base == 'qiskit.ibmq':
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            if params['token']:
                self.token = params['token']
                self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits*self.correction, ibmqx_token= self.token)
            else:
                raise Exception("Token de acceso no encontrado")
        ## Simuladores de qiskit
        elif self.base == "qiskit.aer":
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits*self.correction)
        ## Simuladores de pennylane
        else:
            self.device= qml.device(self.base, wires=self.qubits*self.correction)
        return
    
    def set_node(self, params) -> None:
        if params["pattern"]:
            self.pattern = params["pattern"]
        
        if params['repetitions']:
            self.repetition = params['repetitions']
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return
    

    '''
    Funciones para la construccion del ansatz
    '''
    def single_rotation(self, params, ):
        for i in range(0, self.qubits):
            for j in range(self.correction):
                qml.RY(params[i], wires=[self.correction*i+j])
        return

    def non_local_gates(self):
        ## Compuertas en orden normal
        if self.pattern == 'chain':
            for i in range(0, self.qubits-1):
                for j in range(self.correction):
                    qml.CNOT(wires=[self.correction*i+j, self.correction*(i+1)+j])

        elif self.pattern == 'ring':
            if self.qubits == 2:
                for j in range(self.correction):
                    qml.CNOT(wires=[j, self.correction+j])
            else:
                for i in range(0, self.qubits-1):
                    for j in range(self.correction):
                        qml.CNOT(wires=[self.correction*i+j, self.correction*(i+1)+j])

                for j in range(self.correction):
                    qml.CNOT(wires=[self.correction*(self.qubits-1)+j, j])  
            
        elif self.pattern == 'all_to_all':
            if self.qubits == 2:
                for j in range(self.correction):
                    qml.CNOT(wires=[j, self.correction+j])
            else:
                for i in range(0, self.qubits-1):
                    for k in range(i+1, self.qubits):
                        for j in range(self.correction):
                            qml.CNOT(wires=[self.correction*i+j, self.correction*k +j])
        return
    
    
    def circuit(self, theta, state):
        qml.BasisState(state, wires=range(self.qubits*self.correction))
        rotation_number = self.qubits
        for k in range(0, self.repetition):
            params = theta[k*rotation_number:(k+1)*rotation_number]
            self.single_rotation(params)
            self.non_local_gates(0)
            
        return qml.probs(wires=[i for i in range(self.qubits) ])