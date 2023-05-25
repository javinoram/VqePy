import pennylane as qml

class given_ansatz():
    def circuit(self, theta, obs):
        pass

    device= qml.device("default.qubit", wires=0)
    node = qml.QNode(circuit, device, interface="autograd")
    qubits = 0
    singles=  []
    doubles= []
    repetition = 0
    hf_state = None

    def set_device(self, params) -> None:
        self.device= qml.device(params['backend'], wires=self.qubits)
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
        for _ in range(0, self.repetition):
            for i, term in enumerate(self.singles):
                qml.SingleExcitation(theta[0][i], wires=term)

            for i, term in enumerate(self.doubles):
                qml.DoubleExcitation(theta[1][i], wires=term)
        return qml.expval(obs)

    def draw_circuit(self):
        number = (len(self.singles)+len(self.doubles))*self.repetition
        auxtheta = [i for i in range(number)]
        auxtheta = [auxtheta[:len(self.singles)*self.repetition], auxtheta[len(self.singles)*self.repetition:]]
        return qml.draw(self.circuit)(auxtheta)
    

class spin_ansatz():
    def circuit(self, theta, obs, pauli):
        pass

    backend = "default.qubit"
    device= qml.device("default.qubit", wires=0)
    node = qml.QNode(circuit, device, interface="autograd")
    rotation_set = ["Y"]
    nonlocal_set = ["CX"]
    pattern = "chain"
    qubits = 0
    correction = 1
    repetition = 0
    shots = 1000

    def set_device(self, params) -> None:
        self.backend = params['backend']
        self.device= qml.device(params['backend'], wires=self.qubits)
        return

    def set_hiperparams_circuit(self, params) -> None:
        self.repetition = params['repetitions']
        self.shots = params['shots']
        self.device= qml.device(self.backend, wires=self.qubits*self.correction, shots = int(self.shots))
        return
    
    def set_node(self, params) -> None:
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

    def circuit(self, theta, obs, pauli):
        qml.BasisState([0 for _ in range(self.qubits*self.correction)], wires=range(self.qubits*self.correction))
        rotation_number = self.qubits*len(self.rotation_set)
        for i in range(0, self.repetition):
            self.single_rotation(theta[i*rotation_number:(i+1)*rotation_number])
            self.non_local_gates()

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
    

    def draw_circuit():
        return
    

class spin05_ansatz():
    def circuit(self, theta, obs, state):
        pass

    backend = "default.qubit"
    device= qml.device("default.qubit", wires=0)
    node = qml.QNode(circuit, device, interface="autograd")
    pattern = "chain"
    qubits = 0
    repetition = 0

    def set_device(self, params) -> None:
        self.backend = params['backend']
        self.device= qml.device(params['backend'], wires=self.qubits)
        return

    def set_hiperparams_circuit(self, params) -> None:
        self.repetition = params['repetitions']
        self.shots = params['shots']
        self.device= qml.device(self.backend, wires=self.qubits)
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
        #return qml.expval(obs.hamiltonian(wire_order=range(self.qubits)))
    

    def draw_circuit():
        return