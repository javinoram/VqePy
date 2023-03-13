import numpy as np
import plotly as pt
import pandas as pd
from Ansatzs import *
from VQEclass import *

'''
Open and read parameter file
'''
params = {
    'hamiltonian_terms': [(('XXI', np.sqrt(2))), (('ZZI', np.sqrt(2)))],
    'number_qubits': 3,
    'number_ansatz_repetition': 3,
    'backend': Aer.get_backend('qasm_simulator'),
    'optimization_alg_params': {'tol': 1e-06, 'maxiter': 1e3},
    'gyromagnetic_factor' : 2.0,
    'hamiltonian_vars' : {},
    'optimization_method': 'COBYLA'
}

'''
Simulation enviroment
dict keys: <Observable>_<Ground state or All system>
    inside dict keys <type_of_variable>, <minimun_value>, <minimun_value>, <number_of_point>
'''
if 'magnetizacion_gs' in params['hamiltonian_vars']:
    val_min = params['hamiltonian_vars']['magnetizacion_gs']['minimun_value']
    val_max = params['hamiltonian_vars']['magnetizacion_gs']['maximun_value']
    number_of_points = params['hamiltonian_vars']['magnetizacion_gs']['number_of_points']
    grid = np.linspace(val_min, val_max, number_of_points)
    #for h in grid:

