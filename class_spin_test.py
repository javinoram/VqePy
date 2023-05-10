from classes.variational import *
from classes.ansatz import *
from classes.global_func import *
from classes.hamiltonian import *

params_hamiltonian = {
    "hamiltonian_list": [(("ZZI", -1)), (("IZZ", -1)), (("ZIZ", -1))],
    "spin": 0.5
}

params_ansatz = {
    'repetition': 3,
    'ansatz_pattern': "ring",
}

params_alg = {
    'backend': "default.qubit",
    'optimization_alg_params': {'tol': 1e-6, 'maxiter': 1000},
    'optimization_method': 'COBYLA'
}

spin = spin_hamiltonian(params_hamiltonian, params_alg)
spin.init_ansatz(params_ansatz)

number = sum(spin.number_params)
print("Ground state")
theta = np.array( [np.random.randint(-300, 300) / 100  for _ in range(number)])
print( spin.ground_state_calculation(theta) )

print("Ground state numpy")
sz = np.array([[1,0],[0,-1]])
si = np.array([[1,0],[0,1]])
h = -1*( np.kron(sz, np.kron(sz,si)) + np.kron(si, np.kron(sz,sz)) + np.kron(sz, np.kron(si,sz)) )
ee = np.linalg.eigvals(h)
print(ee)