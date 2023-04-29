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
                qml.SingleExcitation(theta[0][0], wires=term)

            for i, term in enumerate(self.doubles):
                qml.DoubleExcitation(theta[1][i], wires=term)
        return qml.expval(obs)

    def draw_circuit(self):
        number = (len(self.singles)+len(self.doubles))*self.repetition
        auxtheta = [i for i in range(number)]
        auxtheta = [auxtheta[:len(self.singles)*self.repetition], auxtheta[len(self.singles)*self.repetition:]]
        return qml.draw(self.circuit)(auxtheta)
    

class spin_ansatz():
    def circuit(self, theta, obs):
        pass

    backend = "default.qubit"
    device= qml.device("default.qubit", wires=0)
    node = qml.QNode(circuit, device, interface="autograd")
    rotation_set = ["Y"]
    nonlocal_set = ["CRZ"]
    pattern = "chain"
    qubits = 0
    correction = 1
    repetition = 0
    shots = 0

    def set_device(self, params) -> None:
        self.backend = params['backend']
        self.device= qml.device(params['backend'], wires=self.qubits)
        return

    def set_hiperparams_circuit(self, params) -> None:
        self.repetition = params['repetition']
        self.shots = params['shots']
        self.device= qml.device(self.backend, wires=self.qubits*self.correction, shots = self.shots)
        return
    
    def set_node(self, params) -> None:
        self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return
    

    def single_rotation(self, params):
        for i in range( 0, self.qubits):
            for j in range(self.correction):
                #qml.RZ(params[0], wires=[self.correction*i+j])
                qml.RY(params[0], wires=[self.correction*i+j])
                #qml.RX(params[2], wires=[self.correction*i+j])
        return

    def non_local_gates(self, theta):
        if self.pattern == 'chain':
            for i in range(0, self.qubits-1):
                for j in range(self.correction):
                    qml.CRX(theta[i], [self.correction*i+j, self.correction*(i+1)+j])
        if self.pattern == 'ring':
            if self.qubits == 2:
                for j in range(self.correction):
                    qml.CRX(theta[i], [j, self.correction+j])
            else:
                for i in range(0, self.qubits-1):
                    for j in range(self.correction):
                        qml.CRX(theta[i], [self.correction*i+j, self.correction*(i+1)+j])
        return

    def circuit(self, theta, obs):
        qml.BasisState([0 for _ in range(self.qubits*self.correction)], wires=range(self.qubits*self.correction))
        rotation_number = self.qubits*len(self.rotation_set)
        nonlocal_number = self.qubits-1
        for i in range(0, self.repetition):
            self.single_rotation(theta[0][i*rotation_number:(i+1)*rotation_number])
            self.non_local_gates(theta[1][nonlocal_number*i:nonlocal_number*(i+1)])

        aux = []
        for w in obs:
            for i in range(self.correction):
                aux.append( self.correction*w + i)
        return qml.counts(wires=aux)
    

    def draw_circuit():
        return