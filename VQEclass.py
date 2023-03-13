from Ansatzs import *
import scipy as sc

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
        general_cost_function = lambda theta: self.cost_function_VQE(theta) + self.cost_function_magnetic_field(theta, h)
        xs = sc.optimize.minimize(general_cost_function, theta, method=self.optimization_method,
                                  options=self.optimization_alg_params, )['x']
        self.hamiltonian_eigenvalues.append(self.cost_function_VQE(xs))
        self.hamiltonian_eigenvectors.append(xs)
        return self.hamiltonian_eigenvalues[0]
    
    def excited_state_calculation(self, theta: list, number_level: int, h: float) -> float:
        general_cost_function = lambda theta: self.cost_function_VQE(theta) + self.cost_function_magnetic_field(theta, h)
        for _ in range(number_level):
            xs = sc.optimize.minimize(general_cost_function, theta, method=self.optimization_method,
                                    options=self.optimization_alg_params, )['x']
            self.hamiltonian_eigenvalues.append(self.cost_function_VQE(xs))
            self.hamiltonian_eigenvectors.append(xs)
        return self.hamiltonian_eigenvalues

    '''
    Cost functions VQE, VQD ...
    '''
    def cost_function_VQE(self, theta: list) -> float:
        func_val = 0.0
        for i, hamiltonian_term in enumerate(self.hamiltonian_list_index):
            circuit = hardware_efficient_ansatz(theta, self.number_qubits, 2, self.number_ansatz_repetition)
            for j, component in enumerate(hamiltonian_term):
                if 'X' == component[0]:
                    circuit.h( component[1] )
                
                if 'Y' == component[0]:
                    circuit.sdg( component[1] )
                    circuit.h( component[1] )
                circuit.measure( [component[1]], [j])

            job = execute( circuit, self.backend, shots=self.shots)
            result = job.result().get_counts()
            for data_row in result:
                if data_row == '00' or data_row == '11':
                    func_val += self.hamiltonian_terms[i][1]*result[data_row]/self.shots
                else:
                    func_val -= self.hamiltonian_terms[i][1]*result[data_row]/self.shots
        return func_val
    
    def cost_function_VQD(self, theta: list) -> float:
        func_val = self.cost_function_VQE( theta )
        for previus_theta in self.hamiltonian_eigenvectors:

            overlap_circuit = hardware_efficient_ansatz_overlap(theta, previus_theta, self.number_qubits, 2, self.number_ansatz_repetition)
            overlap_circuit.measure([ i for i in range(self.number_qubits)], [ i for i in range(self.number_qubits)])
            
            job = execute( overlap_circuit, self.backend, shots=self.shots)
            result = job.result().get_counts()
            if '000' in result:
                func_val += 50*result['000']/self.shots
        return func_val

    def cost_function_magnetic_field(self, theta: list, h: float) -> float:
        func_val = 0.0
        for qubit_index in range(self.number_qubits):
            circuit = hardware_efficient_ansatz(theta, self.number_qubits, 1, self.number_ansatz_repetition)
            circuit.measure([qubit_index], [0])
            
            job = execute( circuit, self.backend, shots=self.shots)
            result = job.result().get_counts()
            if '0' in result:
                func_val += -h*self.nu_b*self.gyromagnetic_factor*result['0']/self.shots
            else:
                func_val -= -h*self.nu_b*self.gyromagnetic_factor*result['1']/self.shots
        return func_val