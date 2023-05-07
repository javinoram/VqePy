from classes.variational import *
from classes.ansatz import *
from classes.global_func import *
from classes.hamiltonian import *


params_hamiltonian = {
    'symbols': ["H", "H"],
    #'coordinates': np.array([0.0, 0.0, -0.75, 0.0, 0.0, 0.75]),
    'coordinates': np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614]),
    'charge': 0,
    'mult': 1,
    'basis': "sto-3g",
    'method': "dhf",
}

params_ansatz = {
    'repetition': 3,
    'electrons': 2,
}

params_alg = {
    'backend': "default.qubit",
    'optimization_alg_params': {'tol': 1e-6, 'maxiter': 1000},
    'optimization_method': 'COBYLA'
}

h2 = electronic_hamiltonian(params_hamiltonian, params_alg)
h2.init_ansatz(params_ansatz)

number = sum(h2.number_params)

print("Structure")
theta = np.array( [np.random.randint(-300, 300) / 100  for _ in range(number)])
print( h2.structure_calculation(theta) )

print("Ground state")
theta = np.array( [np.random.randint(-300, 300) / 100  for _ in range(number)])
print( h2.ground_state_calculation(theta) )