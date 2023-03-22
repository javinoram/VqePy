from Ansatzs import *
from VQEclass import *

params = {
    'hamiltonian_terms': [(('XXI', -np.sqrt(2))), (('ZZI', -np.sqrt(2)))],
    'number_qubits': 3,
    'number_ansatz_repetition': 4,
    'backend': Aer.get_backend('qasm_simulator'),
    'optimization_alg_params': {'tol': 1e-06, 'maxiter': 1e3},
    'gyromagnetic_factor' : 2.0,
    'hamiltonian_vars' : {},
    'optimization_method': 'COBYLA'
}

simulation_object = hamiltonian(params)
x = np.linspace(0,1,3)*1000
for i in x:
    print(simulation_object.ground_state_calculation( [0.0 for _ in range(12)], -i ))
#print(simulation_object.excited_state_calculation( [0.0 for i in range(15)], 4, 100.0 ))


Si = np.array([[1,0],[0,1]]) 
Sx = np.array([[0,1],[1,0]])
Sy = np.array([[0,-1j],[1j,0]])
Sz = np.array([[1,0],[0,-1]])


H = -np.sqrt(2)*(np.kron(Sx, np.kron(Sx, Si)) + np.kron(Sz, np.kron(Sz, Si))) 
for i in x:
    H = H - 2.0*5.7883818066e-5*i*(np.kron(Sz, np.kron(Si, Si))+ np.kron(Si, np.kron(Sz, Si))+ np.kron(Si, np.kron(Si, Sz)))
    ee, vv = la.eigh( H )
    print(np.round(ee[:5],5))