from quantumsim.functions.ansatz import *
from quantumsim.functions.min_methods import *
from quantumsim.functions.constans import *
from quantumsim.optimizacion_structure import *
from quantumsim.variational_quantum_eigensolver import *
from quantumsim.variational_quantum_thermalizer import *

from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
import sys
import pandas as pd

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

    '''
    Variational Quantum Eigensolver
        Metodo para todo tipo de hamiltoniano
    '''
if params["method_class"] == "VQE":

    if params["simulation_object"] == "Electronic": 
        try:
            symbols, coordinates = qchem.read_structure(params["hamiltonian_params"]["file"])
        except:
            raise Exception("Error en el archivo .xyz")
        
        object_vqe = variational_quantum_eigensolver_electronic(symbols, coordinates, params["hamiltonian_params"])
        object_vqe.set_device(params["ansatz_params"])
        object_vqe.set_hiperparams_circuit(params["ansatz_params"])
        object_vqe.set_node(params["ansatz_params"])

        rep = params["ansatz_params"]["repetitions"]
        number = 2*rep
        theta = np.array( [0.0  for _ in range(number)], requires_grad=True)             
        
    elif params["simulation_object"] == "Spin":

        object_vqe = variational_quantum_eigensolver_spin(params["hamiltonian_params"])
        object_vqe.set_device(params["ansatz_params"])
        object_vqe.set_node(params["ansatz_params"])

        rep = params["ansatz_params"]["repetitions"]
        number = (object_vqe.qubits)*rep
        theta = np.array( [np.random.randint(314)/100.0  for _ in range(number)], requires_grad=True)
    else:
        raise Exception("Error en el tipo de Hamiltoniano")

    if params["minimizate_method"] == "Gradient":
        energy, theta_evol, theta = gradiend_method_VQE(object_vqe.cost_function, theta, params["minimizate_method_params"])
    elif params["minimizate_method"] == "Scipy":
        energy, theta_evol, theta = scipy_method_VQE(object_vqe.cost_function, theta, params["minimizate_method_params"])
    else:
        raise Exception("Metodo de minimizacion ingresado no esta considerado")
    
    #Guarda toda la evolucion de los parametros en un .csv
    #Guarda los parametros optimos en otro archivo
    #
    data={'step':[i for i in range(len(energy))], 'energy': energy}
    theta_evol = np.array(theta_evol).T
    for i in range(len(theta_evol)):
        data["p"+str(i)] = theta_evol[i]
    Result = pd.DataFrame( data )
    Result.to_csv("VQE"+params["simulation_object"]+".csv")
    
    data = {}
    for i in range(len(theta_evol)):
        data["p"+str(i)] = [theta_evol[i][-1]]
    Result = pd.DataFrame( data )
    Result.to_csv("OptimumVQE"+params["simulation_object"]+".csv")

    '''
    Optimizacion de estructura
        Metodo solo para hamiltonianos electronicos
    '''
elif params["method_class"] == "SO":
    if params["simulation_object"] != "Electronic":
        raise Exception("Solo hamiltonianos electronicos")
    
    try:
        symbols, coordinates = qchem.read_structure(params["hamiltonian_params"]["file"])
    except:
        raise Exception("Error en el archivo .xyz")

    object_struc = optimization_structure(symbols, coordinates, params['hamiltonian_params'])
    object_struc.set_device(params["ansatz_params"])
    object_struc.set_hiperparams_circuit(params["ansatz_params"])
    object_struc.set_node(params["ansatz_params"])


    rep = params["ansatz_params"]["repetitions"]
    number = (len(object_struc.singles) + len(object_struc.doubles))*rep
    theta = np.array([0.0  for _ in range(number)])

    theta = np.array( [np.random.randint(314)/100.0  for _ in range(number)] )
    if params["minimizate_method"] == "Gradient":
        energy, theta_evol, theta = gradiend_method_OS(object_struc.cost_function, theta, coordinates, params["minimizate_method_params"], object_struc.grad_x)
    elif params["minimizate_method"] == "Scipy":
        energy, theta_evol, theta = scipy_method_OS(object_struc.cost_function, theta, coordinates, params["minimizate_method_params"])
    else:
        raise Exception("Metodo de minimizacion ingresado no esta considerado")
    
    data={'step':[i for i in range(len(energy))], 'energy': energy}
    theta_evol = np.array(theta_evol).T
    for i in range(len(theta_evol)):
        if i < len(coordinates):
            data["x"+str(i)] = theta_evol[i]
        else:
            data["p"+str(i-len(coordinates))] = theta_evol[i]
    Result = pd.DataFrame( data )
    Result.to_csv("SO"+params["simulation_object"]+".csv")

    data = {}
    for i in range(len(theta_evol)):
        if i < len(coordinates):
            data["x"+str(i)] = [theta_evol[i][-1]]
        else:
            data["p"+str(i-len(coordinates))] = [theta_evol[i][-1]]
    Result = pd.DataFrame( data )
    Result.to_csv("OptimumSO"+params["simulation_object"]+".csv")

else:
    raise Exception("Clase del metodo no valido")