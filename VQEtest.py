from ansatzs import *
from vqe_class import *
import scipy as sc

params = {
    'hamiltonian_terms': [ (('X', -np.sqrt(2))) ],
    'number_qubits': 1,
    'number_ansatz_repetition': 5,
    'backend': Aer.get_backend('qasm_simulator'),
    'optimization_alg_params': {'tol': 1e-06, 'maxiter': 2*(1e3)},
    'gyromagnetic_factor' : 2.0,
    'hamiltonian_vars' : {},
    'optimization_method': 'COBYLA',
    'spin': 1
}

simulation_object = hamiltonian(params)
#xs, val = simulation_object.thermal_state_calculation( [0.0 for _ in range(3*7 + 3)] )
xs, val = simulation_object.ground_state_calculation( [0.0 for _ in range(2*2*5 + 2)], 0.0 )
print(val)
#Si = np.array([[1,0],[0,1]]) 
#Sx = np.array([[0,1],[1,0]])
#Sy = np.array([[0,-1j],[1j,0]])
#Sz = np.array([[1,0],[0,-1]])

Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]])
Sx = (1/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]])

boltz = 8.6173e-5
t = 100.0


#H = -np.sqrt(2)*(np.kron(Sz, Si) + np.kron(Sz, Sz) ) 
#H=-np.sqrt(2)*np.kron(Sz,Sz)
H=-np.sqrt(2)*Sx
#for i in x:
#    H = H - 2.0*5.7883818066e-5*i*(np.kron(Sz, np.kron(Si, Si))+ np.kron(Si, np.kron(Sz, Si))+ np.kron(Si, np.kron(Si, Sz)))
ee, vv = la.eigh( H )
print(ee[:5])
#exp_matrix = sc.linalg.expm(-H/(boltz*t))/np.trace(sc.linalg.expm(-H/(boltz*t)))
#print(np.round(exp_matrix, 3))