from ansatzs import *
#from physics_formulas import *


class hamiltonian():
    #Class variables
    node = None
    dev = None
    hamiltonian_object = None
    nu_b = 5.7883818066e-5
    k_b = 8.617333262e-5
    gyromagnetic_factor: float = 0.0
    spin = 0

    #Circuit params
    number_qubits: int = 0
    number_ansatz_repetition: int = 0

    #Cicuit execution
    backend = None
    shots = 2**10

    #Optimization function params
    optimization_method = "" 
    optimization_alg_params: dict = {}


    '''
    Class inicialization
    '''
    def __init__(self, params) -> None:
        self.gyromagnetic_factor = params['gyromagnetic_factor']
        self.number_qubits = params['number_qubits']
        self.number_ansatz_repetition = params['number_ansatz_repetition']
        self.backend = params['backend']
        self.hamiltonian_vars = params['hamiltonian_vars']
        self.optimization_alg_params = params['optimization_alg_params']
        self.optimization_method = params['optimization_method']
        self.spin = params['spin']
        
        ham_matrix = {}
        for term in params['hamiltonian_terms']:
            aux = {}
            for i in range(self.number_qubits):
                aux[i] = term[0][i]
            ham_matrix[qml.pauli.PauliWord(aux)] = term[1] 
        self.hamiltonian_object = qml.pauli.PauliSentence(ham_matrix).hamiltonian([i for i in range(self.number_qubits)])

        #self.dev = qml.device('qiskit.aer', wires=self.number_qubits, backend='simulator_statevector', shots=self.shots)
        self.dev = qml.device(self.backend, wires=self.number_qubits)
        self.node = qml.QNode(self.quantum_circuit, self.dev, interface="autograd")
        pass

    
    '''
    Optimization functions
    '''
    def ground_state_calculation(self, theta: list, state) -> float:
        general_cost_function = lambda theta: self.cost_function_VQE(theta, state)
        xs = sc.optimize.minimize(general_cost_function, theta, method=self.optimization_method,
                                  options=self.optimization_alg_params, )['x']
        return xs, self.cost_function_VQE(xs, state)
    
    def thermal_state_calculation(self, theta: list, T: float) -> float:
        general_cost_function = lambda theta: self.cost_function_VQT(theta, np.divide(1.0, T))
        xs = sc.optimize.minimize(general_cost_function, theta, method=self.optimization_method,
                                  options=self.optimization_alg_params, )['x']
        return xs, self.cost_function_VQT(xs, np.divide(1.0, T))

    '''
    Cost functions VQE, VQD ...
    '''
    def cost_function_VQE(self, theta: list, state: list) -> float:

        ansatz_params_1 = theta[self.number_qubits : ((self.number_ansatz_repetition + 1) * self.number_qubits)]
        ansatz_params_2 = theta[((self.number_ansatz_repetition + 1) * self.number_qubits) :]

        coupling = np.split(ansatz_params_1, self.number_ansatz_repetition)
        split = np.split(ansatz_params_2, self.number_ansatz_repetition)
        rotation = []
        for s in split:
            rotation.append(np.split(s, 3))

        ansatz_params = [rotation, coupling]
        func_val = self.node( rotation_params = ansatz_params[0], coupling_params = ansatz_params[1], 
                               system_object= self.hamiltonian_object, sample=state )
        return func_val
    
    def cost_function_VQT(self, theta: list, beta: float) -> float:
        dist_params = theta[0:self.number_qubits]
        ansatz_params_1 = theta[self.number_qubits : ((self.number_ansatz_repetition + 1) * self.number_qubits)]
        ansatz_params_2 = theta[((self.number_ansatz_repetition + 1) * self.number_qubits) :]

        coupling = np.split(ansatz_params_1, self.number_ansatz_repetition)
        split = np.split(ansatz_params_2, self.number_ansatz_repetition)
        rotation = []
        for s in split:
            rotation.append(np.split(s, 3))

        ansatz_params = [rotation, coupling]
        parameters =  [dist_params, ansatz_params]

        dist_params = parameters[0]
        ansatz_params = parameters[1]
        distribution = prob_dist(dist_params)
        combos = itertools.product([0, 1], repeat=self.number_qubits)
        list_states = [list(c) for c in combos]

        cost = 0
        for state in list_states:
            result = self.node( rotation_params = ansatz_params[0], coupling_params = ansatz_params[1], 
                               system_object= self.hamiltonian_object, sample=state )
            for j in range(0, len(state)):
                result = result * distribution[j][state[j]]
            cost += result

        entropy = calculate_entropy(distribution)
        func_val = beta*cost - entropy
        return func_val
    

    def quantum_circuit(self, rotation_params, coupling_params, system_object, sample=None):
        qml.BasisStatePreparation(sample, wires=range(self.number_qubits))
        for i in range(0, self.number_ansatz_repetition):
            single_rotation(rotation_params[i], range(self.number_qubits))
            qml.broadcast(
                unitary=qml.CRX,
                pattern="ring",
                wires=range(self.number_qubits),
                parameters=coupling_params[i]
            )
        return qml.expval(system_object)
    
    '''
    Useful functions to study physics object
    '''
    def get_observable(self, theta:list, observable, T=None) -> float:
        dist_params = theta[0:self.number_qubits]
        ansatz_params_1 = theta[self.number_qubits : ((self.number_ansatz_repetition + 1) * self.number_qubits)]
        ansatz_params_2 = theta[((self.number_ansatz_repetition + 1) * self.number_qubits) :]

        coupling = np.split(ansatz_params_1, self.number_ansatz_repetition)
        split = np.split(ansatz_params_2, self.number_ansatz_repetition)
        rotation = []
        for s in split:
            rotation.append(np.split(s, 3))

        ansatz_params = [rotation, coupling]
        parameters =  [dist_params, ansatz_params]

        dist_params = parameters[0]
        ansatz_params = parameters[1]

        if observable=="specific-heat":
            return self.SpecificHeat(ansatz_params, dist_params, T)
        else:
            return self.Enthalpy(ansatz_params, dist_params)
    
    def SpecificHeat(self, ansatz_params, dist_params, T) -> float:
        cost = 0
        distribution = prob_dist(dist_params)
        combos = itertools.product([0, 1], repeat=self.number_qubits)
        list_states = [list(c) for c in combos]

        expH = 0.0
        expHH = 0.0
        for state in list_states:
            result = self.node( rotation_params = ansatz_params[0], coupling_params = ansatz_params[1], 
                system_object= self.hamiltonian_object, sample=state )
            result2 = result*result
            for j in range(0, len(state)):
                result = result * np.sqrt(distribution[j][state[j]])
                result2 = result2* np.sqrt(distribution[j][state[j]])
            expH += result
            expHH += result2
        return np.divide(expHH-expH, T*T*self.k_b)
    
    def Enthalpy(self, ansatz_params, dist_params) -> float:
        cost = 0
        distribution = prob_dist(dist_params)
        combos = itertools.product([0, 1], repeat=self.number_qubits)
        list_states = [list(c) for c in combos]
        for state in list_states:
            result = self.node( rotation_params = ansatz_params[0], coupling_params = ansatz_params[1], 
                system_object= self.hamiltonian_object, sample=state )
            for j in range(0, len(state)):
                result = result * np.sqrt(distribution[j][state[j]])
            cost += result
        return cost
