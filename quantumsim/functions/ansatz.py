import pennylane as qml
from pennylane import numpy as np


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

        basis_change = ['I' for i in range(self.qubits)]
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
    

    
'''
Hardware Efficient ansatz adaptado para trabajar en el modelo de espines
'''
class spin_ansatz():
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
        self.repetition = params['repetitions']
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return
    
    def single_rotation(self, params):
        for i in range(0, self.qubits):
            for j in range(self.correction):
                qml.RY(params[i], wires=[self.correction*i+j])
        return

    def non_local_gates(self):
        if self.pattern == 'chain':
            for i in range(0, self.qubits-1):
                for j in range(self.correction):
                    qml.CNOT(wires=[self.correction*i+j, self.correction*(i+1)+j])
        if self.pattern == 'ring':
            if self.qubits == 2:
                for j in range(self.correction):
                    qml.CNOT(wires=[j, self.correction+j])
            else:
                for i in range(0, self.qubits-1):
                    for j in range(self.correction):
                        qml.CNOT(wires=[self.correction*i+j, self.correction*(i+1)+j])
        return

    def circuit(self, theta, obs):
        qml.BasisState([0 for _ in range(self.qubits*self.correction)], wires=range(self.qubits*self.correction))
        rotation_number = self.qubits
        for k in range(0, self.repetition):
            params = theta[k*rotation_number:(k+1)*rotation_number]
            self.single_rotation(params)
            self.non_local_gates()

        basis_change = ['I' for _ in range(self.qubits)]
        to_measure = []
        for term in obs:
            aux = []
            for j, string in enumerate(term[1]): 
                if string == 'X':
                    if basis_change[j] == 'I': 
                        for k in range(self.correction):
                            qml.Hadamard(wires=[self.correction*j + k])
                        basis_change[j] = 'X'
                        
                    for k in range(self.correction):
                        aux.append( self.correction*j + k)

                elif string == 'Y':
                    if basis_change[j] == 'I': 
                        for k in range(self.correction):
                            qml.S(wires=[self.correction*j + k])
                            qml.Hadamard(wires=[self.correction*j + k])
                        basis_change[j] = 'Y'
                    
                    for k in range(self.correction):
                        aux.append( self.correction*j + k)

                elif string == 'Z':
                    if basis_change[j] == 'I': 
                        basis_change[j] = 'Z'

                    for k in range(self.correction):
                        aux.append( self.correction*j + k)

                else:
                    pass
            to_measure.append(aux)
        
        return [qml.probs(wires=to) for to in to_measure]
    


'''
Given ansatz adaptado para trabajar en el modelo FermiHubbard
'''
class given_fermihubbard_ansazt():
    def circuit(self, theta, obs, type):
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

        #Estado base
        up_array = [0 for _ in range( int(self.qubits/2) )]
        down_array = [0 for _ in range( int(self.qubits/2) )]
        elec = params['electrons']
        upper_index = 0
        lower_index = 0
        for i in range(len(up_array)):
            if elec >0:
                up_array[upper_index] =  1
                upper_index += 1
                elec -= 1 

            if elec >0:
                down_array[lower_index] =  1
                lower_index += 1
                elec -= 1

            if elec <= 0:
                break 

        self.hf_state = np.concatenate((up_array, down_array), axis=0)

        singles = []
        doubles = []
        for i in range(len(up_array)):
            if up_array[i] == 1:
                for j in range(i, len(up_array)):
                    if up_array[j] == 0:
                        singles.append([i,j])

            if down_array[i] == 1:
                for j in range(i, len(down_array)):
                    if down_array[j] == 0:
                        singles.append([i+len(down_array),j+len(down_array)])
            
            if up_array[i] == 1 and down_array[i] == 1:
                for j in range(i, len(down_array)):
                    if up_array[j] == 0 and down_array[j] == 0:
                        doubles.append([i, i+ len(down_array),j, j+len(down_array)])

        self.singles = singles
        self.doubles = doubles
        return
    
    def set_node(self, params) -> None:
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return

    def circuit(self, theta, obs, type, pauli):
        qml.BasisState(self.hf_state, wires=range(self.qubits))
        for i in range(0, self.repetition):
            for term in self.singles:
                qml.SingleExcitation(theta[0][i], wires=term)

            for term in self.doubles:
                qml.DoubleExcitation(theta[1][i], wires=term)

        if type == 'U':
            return qml.probs(wires=obs)
        
        else:
            if 'X' in pauli:
                for i in obs:
                    qml.Hadamard(wires=i)
            if 'Y' in pauli:
                for i in obs:
                    qml.S(wires=i)
                    qml.Hadamard(wires=i)

            return qml.probs(wires=obs)