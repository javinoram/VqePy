from physics.electronic import *

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
    pass

else:
    raise Exception("Hamiltoniano ingreaso no valido")