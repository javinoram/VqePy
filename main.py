from quantumsim.functions.ansatz import *
from quantumsim.functions.min_methods import *
from quantumsim.functions.constans import *
from quantumsim.optimizacion_structure import *
from quantumsim.variational_quantum_eigensolver import *

from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
import sys

import yaml

'''
Regex de ejecución: python3 main.py <parametros .yml>


Lectura de parametros de un archivo YML
'''
with open(sys.argv[len(sys.argv)-1], 'r') as stream:
    try:
        params=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if params["method_class"] == "VQE":
    if params["hamiltonian_params"]["file"] != None:
        symbols, coordinates = qchem.read_structure(params["hamiltonian_params"]["file"])
        object_vqe = variational_quantum_eigensolver_electronic(symbols, coordinates)
    else:
        #AGREGAR CREACION DE ESPINES Y HUBBARD
        pass
    
    object_vqe.set_device(params["ansatz_params"])
    object_vqe.set_hiperparams_circuit(params["ansatz_params"])
    object_vqe.set_node(params["ansatz_params"])

    rep = params["ansatz_params"]["repetitions"]
    number = (len(object_vqe.singles) + len(object_vqe.doubles))*rep
    theta = np.array( [np.random.randint(314)/100.0  for _ in range(number)] )



    energy, theta = gradiend_method_VQE(object_vqe.cost_function, theta, params["minimizate_method_params"])

elif params["method_class"] == "SO":
    if params["hamiltonian_params"]["file"] != None:
        symbols, coordinates = qchem.read_structure(params["hamiltonian_params"]["file"])
        object_struc = variational_quantum_eigensolver_electronic(symbols, coordinates)
    else:
        #AGREGAR CREACION DE ESPINES Y HUBBARD
        pass

    object_struc.set_device(params["ansatz_params"])
    object_struc.set_hiperparams_circuit(params["ansatz_params"])
    object_struc.set_node(params["ansatz_params"])

    rep = params["ansatz_params"]["repetitions"]
    number = (len(object_struc.singles) + len(object_struc.doubles))*rep
    theta = np.array([0.0  for _ in range(number)])
    energy, theta, x = gradiend_method_OS(object_struc.cost_function, theta, coordinates, params, object_struc.grad_x)



