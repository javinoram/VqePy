from pennylane import qchem
from classes.ansatz import *


class electronic_hamiltonian(Given_ansatz):
    hamiltonian_object = None
    number_qubits = None
    coordinates = None
    symbols = None
    extra = None

    dev = None
    node = None
    optimization_method = ""
    optimization_alg_params = {}

    def __init__(self, params, params_alg) -> None:
        _, qubits = qchem.molecular_hamiltonian(
            symbols = params['symbols'],
            coordinates = params['coordinates'],)
        self.number_qubits = qubits
        self.symbols = params['symbols']
        self.coordinates = params['coordinates']

        self.dev = qml.device(params_alg['backend'], wires=self.number_qubits)
        self.node = qml.QNode(self.given_circuit, self.dev, interface="autograd")
        self.optimization_alg_params = params_alg['optimization_alg_params'],
        self.optimization_method = params_alg['optimization_method'],
        return
        
    def init_ansatz(self, params):
        self.repetition = params['repetition']
        self.hf_state = qml.qchem.hf_state(params['electrons'], self.number_qubits)
        self.singles, self.doubles = qml.qchem.excitations(params['electrons'], self.number_qubits)
        self.number_params = [len(self.singles)*self.repetition, len(self.doubles)*self.repetition]
        return
    
    '''
    Method to execute the minimization algorithms
    '''
    def structure_calculation(self, theta: list):
        theta = np.concatenate((self.coordinates, theta), axis=0)
        xs = sc.optimize.minimize(self.cost_function_structure, theta, method=self.optimization_method[0],
                                options=self.optimization_alg_params[0])['x']
        self.coordinates = xs[:3*len(self.symbols)]
        self.hamiltonian_object, _ = qchem.molecular_hamiltonian(
            symbols = self.symbols,
            coordinates = self.coordinates) 
        return np.round(xs[:3*len(self.symbols)],4)
    
    def ground_state_calculation(self, theta: list):
        xs = sc.optimize.minimize(self.cost_function_VQE, theta, method=self.optimization_method[0],
                                options=self.optimization_alg_params[0])['x']
        return xs, self.cost_function_VQE(xs)
    
    '''
    Cost functions for VQE and optimization structure
    '''
    def cost_function_VQE(self, theta: list) -> float:
        params_1 = theta[ :self.number_params[0]]
        params_2 = theta[self.number_params[0]: ]
        result = self.node( qubits= self.number_qubits, params = [params_1, params_2], hamiltonian=self.hamiltonian_object)
        return result
    
    def cost_function_structure(self, theta):
        params_1 = theta[ :len(self.coordinates)]
        params_2 = theta[len(self.coordinates): ]

        params_21 = params_2[ :self.number_params[0]]
        params_22 = params_2[self.number_params[0]: ]

        hamiltonian, qubits = qchem.molecular_hamiltonian( symbols = self.symbols, coordinates = params_1)
        result = self.node(qubits, [params_21, params_22], hamiltonian)
        return result


class spin_hamiltonian(Spin_ansatz, ):
    hamiltonian_object = None
    hamiltonian_index = []
    number_qubits = None
    spin: int = 0
    correction: int = 0
    shots = 3000

    dev = None
    node = None
    optimization_method = ""
    optimization_alg_params = {}

    def __init__(self, params, params_alg) -> None:
        self.number_qubits = len(params['hamiltonian_list'][0][0])
        self.spin = params['spin']
        self.correction = math.ceil( (int( 2*self.spin+1 ))/2  )
        for term in params['hamiltonian_list']:
            aux = []
            for i, string in enumerate(term[0]):
                if string != 'I': aux.append(i)
            self.hamiltonian_index.append(aux)
        self.hamiltonian_object = params['hamiltonian_list']

        self.dev = qml.device(params_alg['backend'], wires=self.number_qubits*self.correction, shots=self.shots)
        self.node = qml.QNode(self.spin_circuit, self.dev, interface="autograd")
        self.optimization_alg_params = params_alg['optimization_alg_params'],
        self.optimization_method = params_alg['optimization_method'],
        return
    
    def init_ansatz(self, params):
        self.repetition = params['repetition']
        self.ansatz_pattern = params['ansatz_pattern']
        self.number_params = [number_rotation_params(self.number_qubits, self.repetition), 
            number_nonlocal_params(self.ansatz_pattern, self.number_qubits, self.repetition)]
        return
    
    def ground_state_calculation(self, theta: list):
        xs = sc.optimize.minimize(self.cost_function_VQE, theta, method=self.optimization_method[0],
                                options=self.optimization_alg_params[0])['x']
        return xs, self.cost_function_VQE(xs)
    
    def cost_function_VQE(self, theta: list) -> float:
        ansatz_1 = theta[0 :self.number_params[0]]
        ansatz_2 = theta[self.number_params[0]: ]
        result= 0.0
        for i, term in enumerate(self.hamiltonian_index):
            result_term = self.node( qubits= self.number_qubits, correction=self.correction, params=[ansatz_1, ansatz_2], wire = term)
            for _, dict_term in enumerate( conts_spin[ str(self.spin) ]["2"] ):
                if dict_term in result_term:
                    exchange = self.hamiltonian_object[i][1]
                    prob = result_term[dict_term]/self.shots
                    const_state = conts_spin[ str(self.spin) ]["2"][dict_term]
                    result += exchange*prob*const_state
        return result

class hubbard_hamiltonian():
    def __init__(self) -> None:
        pass