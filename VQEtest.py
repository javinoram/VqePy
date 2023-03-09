from vqesimulation.Ansatzs import *
from vqesimulation.VQEclass import *

params = {
    'hamiltonian_terms': [(('XXI', np.sqrt(2))), (('ZZI', np.sqrt(2)))],
    'number_qubits': 3,
    'number_ansatz_repetition': 5,
    'backend': Aer.get_backend('qasm_simulator'),
    'optimization_alg_params': {'tol': 1e-06, 'maxiter': 1e3,}
}
simulation_object = hamiltonian(params)
print(simulation_object.hamiltonian_terms)
print(simulation_object.hamiltonian_list_index)
print(simulation_object.ground_state_calculation( [0.0 for i in range(15)] ))
print(simulation_object.hamiltonian_eigenvectors)


Si = np.array([[1,0],[0,1]]) 
Sx = np.array([[0,1],[1,0]])
Sy = np.array([[0,-1j],[1j,0]])
Sz = np.array([[1,0],[0,-1]])

H = np.sqrt(2)*(np.kron(Sx, np.kron(Sx, Si)) + np.kron(Sz, np.kron(Sz, Si)))
ee, vv = la.eigh( H )
print(np.round(ee[:5],5))