from vqesimulation.Ansatzs import *
import scipy as sc

class hamiltonian():
    #Class variables
    hamiltonian_terms: list = []
    hamiltonian_list_index: list = []
    hamiltonian_eigenvalues = []
    hamiltonian_eigenvectors = []

    #Circuit params
    number_qubits: int = 0
    number_ansatz_repetition: int = 0

    #Cicuit execution
    ansatz_circuit: QuantumCircuit = None
    backend = None
    shots = 2**15
    
    #Optimization function params
    optimization_method = "" 
    optimization_alg_params: dict = {}

    '''
    Class inicialization
    '''
    def __init__(self, params) -> None:
        self.hamiltonian_terms = params['hamiltonian_terms']
        self.number_qubits = params['number_qubits']
        self.number_ansatz_repetition = params['number_ansatz_repetition']
        self.backend = params['backend']

        self.optimization_alg_params = params['optimization_alg_params']

        for operator in params['hamiltonian_terms']:
            auxlist = []
            for j in range(len(operator[0])): 
                if operator[0][j]!='I': auxlist.append( ((operator[0][j], j)))
            self.hamiltonian_list_index.append( auxlist )
        pass

    
    '''
    Optimization functions
    '''
    def ground_state_calculation(self, theta: list) -> float:
        xs = sc.optimize.minimize(self.cost_function_VQE, theta, method='COBYLA',
                                  options=self.optimization_alg_params, )['x']
        self.hamiltonian_eigenvalues.append(self.cost_function_VQE(xs))
        self.hamiltonian_eigenvectors.append(xs)
        return self.hamiltonian_eigenvalues[0]
    
    def excited_state_calculation(self, theta: list, number_level: int) -> float:
        func_val = self.cost_function_VQE(theta)
        return func_val

    '''
    Cost functions
    '''
    def cost_function_VQE(self, theta: list) -> float:
        func_val = 0.0
        for i, hamiltonian_term in enumerate(self.hamiltonian_list_index):
            circuit = hardware_efficient_ansatz(theta, self.number_qubits, self.number_ansatz_repetition)
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
        func_val = 0.0
        return func_val