from physics.electronic import *
from physics.spin import *
from physics.functions.min_methods import *

from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
import sys
import pandas as pd

import yaml

with open(sys.argv[len(sys.argv)-1], 'r') as stream:
    try:
        params=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if params["simulation_object"] == "Electronic": 
    try:
        symbols, coordinates = qchem.read_structure(params["hamiltonian_params"]["file"])
    except:
        raise Exception("Error en el archivo .xyz")
    
    if params["observable_params"]['file']:
        theta = pd.read_csv(params["observable_params"]['file']).to_numpy()[0][1:]
    else:
        raise Exception("Archivo de parametros no encontrado")
    
    object_vqe = electronic_hamiltonian(symbols, coordinates, params["hamiltonian_params"])
    object_vqe.set_device(params["ansatz_params"])
    object_vqe.set_hiperparams_circuit(params["ansatz_params"])
    object_vqe.set_node(params["ansatz_params"])

    if params["observable_params"]['observable'] == "Density spin":
        values = object_vqe.density_spin( theta )
    elif params["observable_params"]['observable'] == "Density charge":
        values = object_vqe.density_charge( theta )
    else:
        raise Exception("Observable ingresado no encontrado")
    print(values)

elif params["simulation_object"] == "Spin":     
    if params["observable_params"]['observable'] == "Magnetization GS":
        object_spin = spin_hamiltonian(params["hamiltonian_params"])
        object_spin.set_device(params["ansatz_params"])
        object_spin.set_node(params["ansatz_params"])

        rep = params["ansatz_params"]["repetitions"]
        
        if params["observable_params"]['file']:
            theta = pd.read_csv(params["observable_params"]['file']).to_numpy()[0][1:]
        else:
            raise Exception("Archivo de parametros no encontrado")
        print( object_spin.magnetization(theta) )

    if params["observable_params"]['observable'] == "Specific heat":
        sup = params["observable_params"]['lim_sup']
        inf = params["observable_params"]['lim_inf']
        n = params["observable_params"]['number_points']
        Temperature = np.linspace(inf, sup, n)

        object_vqt = variational_quantum_thermalizer_spin(params["hamiltonian_params"])
        object_vqt.set_device(params["ansatz_params"])
        object_vqt.set_node(params["ansatz_params"])

        rep = params["ansatz_params"]["repetitions"]
        number = (object_vqt.qubits)*rep

        for T in Temperature:
            theta = np.array( [np.random.randint(314)/100.0  for _ in range(number)] )
            dist = np.array( [np.random.randint(100)/100.0  for _ in range(object_vqt.qubits)] )

            if params["minimizate_method"] == "Gradient":
                theta = gradiend_method_VQT(object_vqt.cost_function, theta, dist, 1/T, params["minimizate_method_params"])
            elif params["minimizate_method"] == "Scipy":
                theta = scipy_method_VQT(object_vqt.cost_function, theta, dist, 1/T, params["minimizate_method_params"])
            else:
                raise Exception("Metodo de minimizacion ingresado no esta considerado")

            print( object_vqt.specific_heat(theta, T) )
        pass
    else:
        raise Exception("Observable ingresado no encontrado")
else:
    raise Exception("Hamiltoniano ingreaso no valido")