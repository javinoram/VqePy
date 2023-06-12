import pennylane as qml
from pennylane.templates import ApproxTimeEvolution
from physics.functions.constans import *

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
                    wires=self.qubits, 
                    ibmqx_token= self.token)
            else:
                raise Exception("Token de acceso no encontrado")
        ##Simuladores de qiskit
        elif self.base == "qiskit.aer":
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            
            self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits*self)
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
        self.node = qml.QNode(self.circuit_time, self.device, interface=params['interface'])
        return
    

    def circuit_time(self, theta, obs, time, n, hamiltonian):
        '''Ground state'''
        qml.BasisState(self.hf_state, wires=range(self.qubits))
        for i in range(0, self.repetition):
            for j, term in enumerate(self.singles):
                qml.SingleExcitation(theta[0][i], wires=term)

            for j, term in enumerate(self.doubles):
                qml.DoubleExcitation(theta[1][i], wires=term)

        '''Time evolution'''
        for _ in range(n):
            for coeff, term in hamiltonian:
                if is_identity(term):
                    pass
                else:
                    non_null_index = []
                    for k in range(len(term)):
                        if term[k] != 'I':
                            non_null_index.append(k)

                    '''Initial basis change'''
                    for k in range(len(non_null_index)):
                        if term[k] == 'X':
                            qml.Hadamard(wires=[non_null_index[k]])
                        elif term[k] == 'Y':
                            qml.S(wires=[non_null_index[k]])
                            qml.Hadamard(wires=[non_null_index[k]])
                        else:
                            pass
                    '''Initial CX gates'''
                    for k in range(len(non_null_index)-1):
                        qml.CNOT(wires=[non_null_index[k], non_null_index[k+1]])

                    '''Time parameter'''
                    qml.RZ( (2*time/n)*coeff, non_null_index[-1] )

                    '''Final CX gates'''
                    for k in range(len(non_null_index)-1):
                        qml.CNOT(wires=[non_null_index[k], non_null_index[k+1]])
                    
                    '''Final basis change'''
                    for k in range(len(non_null_index)):
                        if term[k] == 'X':
                            qml.Hadamard(wires=[non_null_index[k]])
                        elif term[k] == 'Y':
                            qml.Hadamard(wires=[non_null_index[k]])
                            qml.S(wires=[non_null_index[k]])
                        else:
                            pass
        return qml.probs(wires=obs[0]), qml.probs(wires=obs[1])

class spin_ansatz():
    def circuit(self, theta, obs, pauli):
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
    shots = 1000

    def set_device(self, params) -> None:
        self.base = params['base']
        self.shots = params['shots']

        ##Maquinas reales
        if self.base == 'qiskit.ibmq':
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            if params['token']:
                self.token = params['token']
                self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits*self.correction, shots = int(self.shots), 
                    ibmqx_token= self.token)
            else:
                raise Exception("Token de acceso no encontrado")
        ##Simuladores de qiskit
        elif self.base == "qiskit.aer":
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits*self.correction, shots = int(self.shots))
        ##Simuladores de pennylane
        else:
            self.device= qml.device(self.base, wires=self.qubits*self.correction, shots = int(self.shots))
        return
    
    def set_node(self, params) -> None:
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return
    
    def ZZ(self, index, t):
        qml.CNOT(wires=index)
        qml.RZ(2*t, wires=index[1])
        qml.CNOT(wires=index)
        return

    def YY(self, index, t):
        qml.S(wires=index)
        qml.H(wires=index)

        qml.CNOT(wires=index)
        qml.RZ(2*t, wires=index[1])
        qml.CNOT(wires=index)

        qml.H(wires=index)
        qml.S(wires=index)
        return

    def XX(self, index, t):
        qml.H(wires=index)

        qml.CNOT(wires=index)
        qml.RZ(2*t, wires=index[1])
        qml.CNOT(wires=index)

        qml.H(wires=index)
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

    def circuit(self, theta, hamiltonian, index_list, obs, pauli, time):
        qml.BasisState([0 for _ in range(self.qubits*self.correction)], wires=range(self.qubits*self.correction))
        rotation_number = self.qubits
        for i in range(0, self.repetition):
            self.single_rotation(theta[i*rotation_number:(i+1)*rotation_number])
            self.non_local_gates()


        for i, term in enumerate(hamiltonian):
            if 'Z' in term[0]:
                self.ZZ(index_list[i], time)
            if 'Y' in term[0]:
                self.YY(index_list[i], time)
            if 'X' in term[0]:
                self.XX(index_list[i], time)


        aux = []
        for w in obs:
            for i in range(self.correction):
                aux.append( self.correction*w + i)

        if 'X' in pauli:
            for i in aux:
                for j in range(self.correction):
                    qml.Hadamard(wires=[self.correction*i+j])
        if 'Y' in pauli:
            for i in aux:
                for j in range(self.correction):
                    qml.S(wires=[self.correction*i+j])
                    qml.Hadamard(wires=[self.correction*i+j])
        return qml.probs(wires=aux)
    
class spin05_ansatz():
    def circuit(self, theta, obs, state):
        pass

    base = ""
    backend = ""
    token = ""

    device= qml.device("default.qubit", wires=0)
    node = qml.QNode(circuit, device, interface="autograd")
    pattern = "chain"
    qubits = 0
    repetition = 0

    def set_device(self, params) -> None:
        self.base = params['base']
        self.shots = params['shots']

        ##Maquinas reales
        if self.base == 'qiskit.ibmq':
            self.backend = params['backend']
            if params['token']:
                self.token = params['token']
                self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits, shots = int(self.shots), 
                    ibmqx_token= self.token)
            else:
                raise Exception("Token de acceso no encontrado")
        ##Simuladores de qiskit
        elif self.base == "qiskit.aer":
            self.backend = params['backend']
            self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits*self, shots = int(self.shots))
        ##Simuladores de pennylane
        else:
            self.device= qml.device(self.base, wires=self.qubits, shots = int(self.shots))
        return
    
    def set_node(self, params) -> None:
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return
    

    def single_rotation(self, params):
        for i in range(0, self.qubits):
            qml.RY(params[i], wires=[i])
        return

    def non_local_gates(self):
        if self.pattern == 'chain':
            for i in range(0, self.qubits-1):
                qml.CNOT(wires=[i, i+1])
        if self.pattern == 'ring':
            if self.qubits == 2:
                qml.CNOT(wires=[0, 1])
            else:
                for i in range(0, self.qubits-1):
                    qml.CNOT(wires=[i, (i+1)])
        return

    def circuit(self, theta, obs, pauli, state):
        qml.BasisState(state, wires=range(self.qubits))
        rotation_number = self.qubits
        for i in range(0, self.repetition):
            self.single_rotation(theta[i*rotation_number:(i+1)*rotation_number])
            self.non_local_gates()
        
        if 'X' in pauli:
            qml.Hadamard(wires=obs)
        if 'Y' in pauli:
            qml.S(wires=obs)
            qml.Hadamard(wires=obs)
        return qml.probs(wires=obs)