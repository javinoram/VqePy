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
    hf_state = []
    state = []

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
        self.hf_state = qml.qchem.hf_state(params['electrons'], self.qubits)
        self.singles, self.doubles = qml.qchem.excitations(params['electrons'], self.qubits)
        return
    
    def set_state(self, state) -> None:
        aux = []
        for i in range(len(state)):
            if state[i] == '1':
                aux.append(1)
            else:
                aux.append(0)
        self.state = aux
        return
    
    def set_node(self, params) -> None:
        self.repetition = params['repetitions']
        self.node = qml.QNode(self.circuit_time, self.device, interface=params['interface'])
        return
    

    def circuit_time(self, theta, obs, time, n, hamiltonian):
        '''Ground state'''
        if len(self.hf_state) != 0:
            qml.BasisState(self.hf_state, wires=range(self.qubits))
            for i in range(0, self.repetition):
                for j, term in enumerate(self.singles):
                    qml.SingleExcitation(theta[0][i], wires=term)

                for j, term in enumerate(self.doubles):
                    qml.DoubleExcitation(theta[1][i], wires=term)
        else:
            qml.BasisState(self.state, wires=range(self.qubits))

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
    state = []
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
    
    def set_node(self, params, state, type) -> None:
        if state != []:
            aux = []
            for i in range(len(state)):
                aux.append( int(state[i], base=10) )

            aux_state = []
            for i in range(self.qubits):
                term = format(aux[i], 'b').zfill(self.correction)
                for s in term:
                    aux_state.append(int(s, base=10))
            self.state = aux_state
        else:
            self.state = []
        if type == 'Time':
            self.node = qml.QNode(self.time_circuit, self.device, interface=params['interface'])
        else:
            self.node = qml.QNode(self.thermal_circuit, self.device, interface=params['interface'])
        return
    
    def ZZ(self, index, t, exchange):
        qml.CNOT(wires=index)
        qml.RZ(2*t*exchange, wires=index[1])
        qml.CNOT(wires=index)
        return

    def YY(self, index, t, exchange):
        for i in index:
            qml.S(wires=i)
            qml.Hadamard(wires=i)

        qml.CNOT(wires=index)
        qml.RZ(2*t*exchange, wires=index[1])
        qml.CNOT(wires=index)

        for i in index:
            qml.Hadamard(wires=i)
            qml.S(wires=i)
        return

    def XX(self, index, t, exchange):
        for i in index:
            qml.Hadamard(wires=i)

        qml.CNOT(wires=index)
        qml.RZ(2*t*exchange, wires=index[1])
        qml.CNOT(wires=index)

        for i in index:
            qml.Hadamard(wires=i)
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

    def time_circuit(self, theta, hamiltonian, time, n):
        if self.state == []:
            qml.BasisState([0 for _ in range(self.qubits*self.correction)], wires=range(self.qubits*self.correction))
            rotation_number = self.qubits
            for i in range(0, self.repetition):
                self.single_rotation(theta[i*rotation_number:(i+1)*rotation_number])
                self.non_local_gates()
        else:
            qml.BasisState(self.state, wires=range(self.qubits*self.correction))
        
        for _ in range(n):
            for coeff, term in hamiltonian:
                index = []
                for i in range(len(term)):
                    if term[i] != 'I':
                        for j in range(self.correction):
                            index.append(self.correction*i + j)

                division = int(len(index)/2)
                q1 = aux[:division]
                q2 = aux[division:]

                if 'Z' in term:
                    for j in range(division):
                        self.ZZ([q1[j], q2[j]], time, coeff/n)
                if 'Y' in term:
                    for j in range(division):
                        self.YY([q1[j], q2[j]], time, coeff/n)
                if 'X' in term:
                    for j in range(division):
                        self.XX([q1[j], q2[j]], time, coeff/n)


        to_measure = []
        for w in range(self.qubits):
            aux = []
            for i in range(self.correction):
                aux.append( self.correction*w + i)
            to_measure.append(aux)
        return [qml.probs(wires=to) for to in to_measure ]
    

    def thermal_circuit(self, theta, obs, state):
        qml.BasisState(state, wires=range(self.qubits*self.correction))
        rotation_number = self.qubits
        for i in range(0, self.repetition):
            self.single_rotation(theta[i*rotation_number:(i+1)*rotation_number])
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
        return [qml.probs(wires=to) for to in to_measure ]