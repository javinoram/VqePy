from ansatzs import *
from physics_formulas import *

class hamiltonian():
    #Class variables
    hamiltonian_terms: list = []
    hamiltonian_list_index: list = []
    hamiltonian_eigenvalues = []
    hamiltonian_eigenvectors = []
    hamiltonian_vars = {}
    nu_b = 5.7883818066e-5
    k_b = 8.617333262e-5
    gyromagnetic_factor: float = 0.0
    spin = 1.0/2

    #Circuit params
    number_qubits: int = 0
    number_ansatz_repetition: int = 0

    #Cicuit execution
    ansatz_circuit: QuantumCircuit = None
    backend = None
    shots = 2**12
    
    #Optimization function params
    optimization_method = "" 
    optimization_alg_params: dict = {}

    '''
    Class inicialization
    '''
    def __init__(self, params) -> None:
        self.hamiltonian_terms = params['hamiltonian_terms']
        self.gyromagnetic_factor = params['gyromagnetic_factor']
        self.number_qubits = params['number_qubits']
        self.number_ansatz_repetition = params['number_ansatz_repetition']
        self.backend = params['backend']
        self.hamiltonian_vars = params['hamiltonian_vars']
        self.optimization_alg_params = params['optimization_alg_params']
        self.optimization_method = params['optimization_method']
        self.spin = params['spin']

        for operator in params['hamiltonian_terms']:
            auxlist = []
            for j in range(len(operator[0])): 
                if operator[0][j]!='I': auxlist.append( ((operator[0][j], j)))
            self.hamiltonian_list_index.append( auxlist )
        pass

    
    '''
    Optimization functions
    '''
    def ground_state_calculation(self, theta: list, h: float) -> float:
        general_cost_function = lambda theta: self.cost_function_VQE(theta, h)
        xs = sc.optimize.minimize(general_cost_function, theta, method=self.optimization_method,
                                  options=self.optimization_alg_params, )['x']
        return xs, self.cost_function_VQE(xs, h)
    
    def thermal_state_calculation(self, theta: list) -> float:
        general_cost_function = lambda theta: self.cost_function_VQT(theta)
        xs = sc.optimize.minimize(general_cost_function, theta, method=self.optimization_method,
                                  options=self.optimization_alg_params, )['x']
        return xs, self.cost_function_VQT(theta)

    '''
    Cost functions VQE, VQD ...
    '''
    def cost_function_VQE(self, theta: list, h: float) -> float:
        func_val = 0.0

        circuit = hardware_efficient_ansatz(self.spin, theta, self.number_qubits, 1, self.number_ansatz_repetition)
        adjust_number = math.ceil((2*self.spin+1)/2.0)
        for i, hamiltonian_term in enumerate(self.hamiltonian_list_index):
            circuit_copy = circuit
            for j, component in enumerate(hamiltonian_term):
                for k in range(adjust_number):
                    if 'X' == component[0]:
                        circuit_copy.h( component[1]*adjust_number+k )
                        
                    if 'Y' == component[0]:
                        circuit_copy.sdg( component[1]*adjust_number+k )
                        circuit_copy.h( component[1]*adjust_number+k )
                    circuit_copy.measure( [component[1]*adjust_number+k], [j*adjust_number+ k])

            job = execute( circuit_copy, self.backend, shots=self.shots)
            result = job.result().get_counts()
            for data_row in result:
                if self.spin ==0.5:
                    if data_row == '00' or data_row == '11':
                        func_val += self.hamiltonian_terms[i][1]*result[data_row]/self.shots
                    else:
                        func_val -= self.hamiltonian_terms[i][1]*result[data_row]/self.shots
                else:
                    if data_row[:adjust_number] == '00':
                        func_val += self.hamiltonian_terms[i][1]*result[data_row]/self.shots
                    elif data_row[adjust_number:] == '10':
                        func_val -= self.hamiltonian_terms[i][1]*result[data_row]/self.shots
            print(func_val, result)
    
        #circuit = hardware_efficient_ansatz(theta, self.number_qubits, 1, self.number_ansatz_repetition)
        #for qubit_index in range(self.number_qubits):
        #    circuit_copy = circuit
        #    circuit_copy.measure([qubit_index], [0])
                
        #    job = execute( circuit_copy, self.backend, shots=self.shots)
        #    result = job.result().get_counts()
        #    if '0' in result:
        #        func_val += h*self.nu_b*self.gyromagnetic_factor*result['0']/self.shots
        #    else:
        #        func_val -= h*self.nu_b*self.gyromagnetic_factor*result['1']/self.shots
        return func_val