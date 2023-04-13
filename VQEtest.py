from ansatzs import *
from vqe_class import *

params = {
    'hamiltonian_terms': [ (('ZZI', 1.0)),(('IZZ', 1.0)), (('ZIZ', 1.0)) ],
    'number_qubits': 3,
    'number_ansatz_repetition': 3,
    'backend': "default.qubit",
    'optimization_alg_params': {'tol': 1e-6, 'maxiter': 600},
    'gyromagnetic_factor' : 2.0,
    'hamiltonian_vars' : {},
    'optimization_method': 'COBYLA',
    'spin': 0.5
}

simulation_object = hamiltonian(params)
number_params = int( params['number_qubits'] * (1 + params['number_ansatz_repetition'] * 4) )

for i in np.linspace(1,10,5):
    print("temperatura: ", i)
    xs, val = simulation_object.thermal_state_calculation( [np.random.randint(-300, 300) / 100  for _ in range((number_params))], 10.0 )
    print( xs[:7] )
    #print( simulation_object.get_observable(xs, "specific-heat", 10.0) )
#xs, val = simulation_object.ground_state_calculation( [np.random.randint(-300, 300) / 100  for _ in range((number_params))], [0,0,0] )
#print(val)

Si = np.array([[1,0],[0,1]]) 
Sx = np.array([[0,1],[1,0]])
Sy = np.array([[0,-1j],[1j,0]])
Sz = np.array([[1,0],[0,-1]])
H = (np.kron(np.kron(Sz,Sz),Si) + np.kron(np.kron(Sz,Si),Sz) + np.kron(np.kron(Si,Sz),Sz))
ee, vv = la.eigh( H )
print(ee)