from ansatzs import *
from classes import *
from pennylane import qchem
import numpy as np

#params_hamiltonian = {
#    'file_name': 'h2.xyz',
#    'charge': 0,
#    'mult': 1,
#    'basis': "sto-3g",
#    'method': "dhf",
#}

si = np.array([[1,0,0], [0,1,0], [0,0,1]])
sz = np.array([[1,0,0], [0,0,0], [0,0,-1]])

h = -2*np.kron(sz,np.kron(sz, si)) -2*np.kron(si,np.kron(sz, sz)) -2*np.kron(sz,np.kron(si, sz))

params_hamiltonian = {
    'hamiltonian_list':[(('ZZI', -2)), (('IZZ', -2)), (('ZIZ', -2))],
    'spin': 1
}

params_ansatz = {
    'repetition': 3,
    'ansatz_pattern': 'all_to_all',
    'rotation_set': ['Z', 'Y', 'X']
}

params_alg = {
    'backend': "default.qubit",
    'optimization_alg_params': {'tol': 1e-6, 'maxiter': 1000},
    'optimization_method': 'COBYLA'
}

VQE = variational_quantum_eigensolver(params_hamiltonian, params_ansatz, params_alg)
spin = 1
correction = math.ceil( (int( 2*spin+1 ))/2  )
print(VQE.number_qubits)
print(VQE.hamiltonian_object)
print(VQE.hamiltonian_index)


number = VQE.number_nonlocal + VQE.number_rotations
theta = np.array( [np.random.randint(-300, 300) / 100  for _ in range(number)])
print( VQE.ground_state_calculation(theta, 2) )

print( np.linalg.eigvals(h)[:4] )