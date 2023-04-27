from ansatzs import *
from pennylane import qchem
from pennylane import numpy as np
#from physics_formulas import *

conts_spin = {"0.5": {"1": { '0':1, '1':-1, },
                      "2": {'00':1, '11':1, '01': -1, '10':-1}
                      },
            "1": {"1": {'00':1, '01':0, '10':-1, '11':0},
                      "2": {'0000':1, '1010':1, '0010': -1, '1000':-1}
                      },
            "1.5": {"1": {'00':3/2, '01':1/2, '10':-1/2, '11':-3/2},
                    "2": {'0000':9/4, '0001':3/4, '0010':-3/4, '0011':-9/4, 
                          '0100':3/4, '0101':1/4, '0110':-1/4, '0111':-3/4,
                          '1000':-3/4, '1001':-1/4, '1010':1/4, '1011':-3/4,
                          '1100':-9/4, '1101':-3/4, '1110':3/4, '1111':9/4}  },
            "2": {"1": {'000':2, '001':1, '010':0, '011':-1, '100':2},
                      },
            "2.5": {"1": {'000':5/2, '001':3/2, '010':1/2, '011':-1/2, '100':-3/2, '101':-5/2},
                      },
            }
'''
class hamiltonian():
    #Class variables
    node = None
    dev = None
    hamiltonian_object = None
    k_b: float = 8.617333262e-2

    #Circuit params
    number_qubits: int = 0
    number_ansatz_repetition: int = 0

    #Circuit execution
    ansatz_pattern = 'chain'
    backend = None


    #Optimization function params
    optimization_method = "" 
    optimization_alg_params: dict = {}

    def __init__(self, params) -> None:
        self.number_ansatz_repetition = params['number_ansatz_repetition']
        self.backend = params['backend']
        self.optimization_alg_params = params['optimization_alg_params']
        self.optimization_method = params['optimization_method']
        self.ansatz_pattern = params['ansatz_pattern']
        pass

    def thermal_state_calculation(self, theta: list, T: float) -> float:
        general_cost_function = lambda theta: self.cost_function_VQT(theta, np.divide(1.0, T))
        xs = sc.optimize.minimize(general_cost_function, theta, method=self.optimization_method,
                                  options=self.optimization_alg_params, )['x']
        return xs, general_cost_function(xs)
    
    def cost_function_VQT(self, theta: list, beta: float) -> float:
        #dist_params = theta[0:self.number_qubits]
        #ansatz_params_1 = theta[self.number_qubits :int( self.number_ansatz_repetition*(self.number_qubits/2)*(self.number_qubits-1))]
        #ansatz_params_2 = theta[int( self.number_ansatz_repetition*(self.number_qubits/2)*(self.number_qubits-1)):]
        
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
            result= self.node( rotation_params = ansatz_params[0], coupling_params = ansatz_params[1], sample=state)
            for j in range(0, len(state)):
                result = result * distribution[j][state[j]]
            cost += result

        entropy = calculate_entropy(distribution)
        func_val = beta*cost - entropy
        return func_val
    
'''

class hamiltonian():
    hamiltonian_object = None
    hamiltonian_index = []
    number_qubits = None
    spin = 0.5

    def __init__(self) -> None:
        pass

    def init_hamiltonian_file(self, file_name, params) -> None:
        symbols, coordinates = qchem.read_structure( file_name )
        hamiltonian, qubits = qchem.molecular_hamiltonian(
            symbols, coordinates,
            charge= params['charge'],
            mult= params['mult'],
            basis= params['basis'],
            method= params['method'])
        self.hamiltonian_object = hamiltonian
        self.number_qubits = qubits
        pass

    def init_hamiltonian_list(self, params) -> None:
        ham_matrix = {}
        self.number_qubits = len(params['hamiltonian_list'][0][0])
        self.spin = params['spin']
        for term in params['hamiltonian_list']:
            aux = []
            for i, string in enumerate(term[0]):
                if string != 'I': aux.append(i)
            self.hamiltonian_index.append(aux)

        if self.spin == 0.5:
            for term in params['hamiltonian_list']:
                aux = {}
                for i in range(self.number_qubits):
                    aux[i] = term[0][i]
                ham_matrix[qml.pauli.PauliWord(aux)] = term[1] 
            self.hamiltonian_object = qml.pauli.PauliSentence(ham_matrix).hamiltonian([i for i in range(self.number_qubits)])
        else:
            self.hamiltonian_object = params['hamiltonian_list']
        pass

class ansatz():
    repetition: int = 0
    ansatz_pattern: str = ""
    number_rotations: int = 0
    number_nonlocal: int = 0
    rotation_set: list = []

    def __init__(self) -> None:
        pass

    def init_ansatz(self, params, qubits) -> None:
        self.repetition = params['repetition']
        self.ansatz_pattern = params['ansatz_pattern']
        self.rotation_set = params['rotation_set']
        self.number_rotations = number_rotation_params(self.rotation_set, qubits, self.repetition)
        self.number_nonlocal = number_nonlocal_params(self.ansatz_pattern, qubits, self.repetition)
        return 

    def single_rotation(self, phi_params, qubits, spin):
        correction = math.ceil( (int( 2*spin+1 ))/2  )
        for i in range( 0, qubits):
            for j in range(correction):
                qml.RZ(phi_params[i][0], wires=[correction*i+j])
                qml.RY(phi_params[i][1], wires=[correction*i+j])
                qml.RX(phi_params[i][2], wires=[correction*i+j])

    def quantum_circuit(self, qubits, spin,  rotation_params, coupling_params, wire, sample=None, system_object=None):
        correction = math.ceil( (int( 2*spin+1 ))/2  )
        qml.BasisState(sample, wires=range(correction*qubits))
        for i in range(0, self.repetition):
            single_rotation(rotation_params[i], qubits, spin)
            #qml.broadcast(
            #    unitary=qml.CRX, pattern=self.ansatz_pattern,
            #    wires=range(qubits), parameters=coupling_params[i]
            #)

        if system_object == None:
            aux = []
            for w in wire:
                for i in range(correction):
                    aux.append( correction*w + i)
            return qml.counts(wires=aux)
        else:
            return qml.expval(system_object)

class variational_quantum_eigensolver(hamiltonian, ansatz):
    dev = None
    node = None
    backend = None
    optimization_method = None
    optimization_alg_params = None

    def __init__(self, params_hamiltonian, params_ansatz, params_alg) -> None:
        if 'file_name' in params_hamiltonian:
            self.init_hamiltonian_file(params_hamiltonian['file_name'], params_hamiltonian)
        else:
            self.init_hamiltonian_list(params_hamiltonian)
        self.init_ansatz(params_ansatz, self.number_qubits)

        correction = math.ceil( (int( 2*self.spin+1 ))/2  )
        self.dev = qml.device(params_alg['backend'], wires=correction*self.number_qubits, shots=1000)
        self.node = qml.QNode(self.quantum_circuit, self.dev, interface="autograd")

        self.optimization_alg_params = params_alg['optimization_alg_params'],
        self.optimization_method = params_alg['optimization_method'],
        pass

    def ground_state_calculation(self, theta: list, electrons: int ) -> float:
        hf = qml.qchem.hf_state(electrons, self.number_qubits)
        general_cost_function = lambda theta: self.cost_function_VQE(theta, hf)
        xs = sc.optimize.minimize(general_cost_function, theta, method=self.optimization_method,
                                options=self.optimization_alg_params)['x']
        return xs, general_cost_function(xs)

    def cost_function_VQE(self, theta: list, state: list) -> float:
        ansatz_params_1 = theta[0 :self.number_nonlocal]
        ansatz_params_2 = theta[self.number_nonlocal: ]
        coupling = np.split(ansatz_params_1, self.repetition)
        split = np.split(ansatz_params_2, self.repetition)
        rotation = []
        for s in split:
            rotation.append(np.split(s, 3))

        if self.spin != 0.5:
            result= 0.0
            for i, term in enumerate(self.hamiltonian_index):
                result_term = self.node( qubits= self.number_qubits, spin=self.spin, rotation_params = rotation, coupling_params = coupling, wire = term, sample=state )
                for _, dict_term in enumerate( conts_spin[ str(self.spin) ]["2"] ):
                    if dict_term in result_term:
                        exchange = self.hamiltonian_object[i][1]
                        prob = result_term[dict_term]/1000.0
                        const_state = conts_spin[ str(self.spin) ]["2"][dict_term]
                        result += exchange*prob*const_state
        else:
            result = self.node( qubits= self.number_qubits, spin=self.spin, rotation_params = rotation, 
                coupling_params = coupling, wire = range(self.number_qubits), sample=state, system_object= self.hamiltonian_object )
        return result