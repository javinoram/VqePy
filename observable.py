from physics.electronic import *
from physics.spin import *
from physics.functions.min_methods import *

from pennylane import numpy as np
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
        object_vqe = electronic_hamiltonian(symbols, coordinates, params["hamiltonian_params"])
        object_vqe.set_device(params["ansatz_params"])
        object_vqe.set_hiperparams_circuit(params["ansatz_params"])
        object_vqe.set_node(params["ansatz_params"])

        n = params["observable_params"]['number_points']
        Time = [ params["observable_params"]['delta']*i for i in range(n) ]
        density_values = []

        if params["observable_params"]['observable'] == "Density spin":
            for t in Time:
                value = object_vqe.density_spin( theta, t, 7)
                density_values.append( value )
        elif params["observable_params"]['observable'] == "Density charge":
            for t in Time:
                value = object_vqe.density_charge( theta, t, 7)
                density_values.append( value )
        else:
            raise Exception("Observable ingresado no encontrado")


    elif params["observable_params"]['state']:
        state = params["observable_params"]['state']
        object_vqe = electronic_hamiltonian(symbols, coordinates, params["hamiltonian_params"])
        object_vqe.set_device( params["ansatz_params"] )
        object_vqe.set_state( params["observable_params"]['state'] )
        object_vqe.set_node( params["ansatz_params"] )

        n = params["observable_params"]['number_points']
        Time = [ params["observable_params"]['delta']*i for i in range(n) ]
        density_values = []

        if params["observable_params"]['observable'] == "Density spin":
            for t in Time:
                value = object_vqe.density_spin( [], t, 7 )
                density_values.append( value )
        elif params["observable_params"]['observable'] == "Density charge":
            for t in Time:
                value = object_vqe.density_charge( [], t, 7 )
                density_values.append( value )
        else:
            raise Exception("Observable ingresado no encontrado")

    else:
        raise Exception("Archivo de parametros optimos o estado no se encontraron")
    
    Result = pd.DataFrame( {'time': Time, 'values':density_values} )
    Result.to_csv("DensityValues.csv")
    


elif params["simulation_object"] == "Spin":
    if params["observable_params"]['observable'] == "Magnetization":
        if params["observable_params"]['file']:
            theta = pd.read_csv(params["observable_params"]['file']).to_numpy()[0][1:]
            object_vqe = spin_hamiltonian(params["hamiltonian_params"])
            object_vqe.set_device(params["ansatz_params"])
            object_vqe.set_node(params["ansatz_params"], [], 'Time')

            n = params["observable_params"]['number_points']
            Time = [ params["observable_params"]['delta']*i for i in range(n) ]
            Magnetization = []

            for t in Time:
                value = object_vqe.magnetization( [], t, 7 )
            


        elif params["observable_params"]['state']:
            state = params["observable_params"]['state']
            object_vqe = spin_hamiltonian(params["hamiltonian_params"])
            object_vqe.set_device( params["ansatz_params"] )
            object_vqe.set_node( params["ansatz_params"], state, 'Time')

            n = params["observable_params"]['number_points']
            Time = [ params["observable_params"]['delta']*i for i in range(n) ]
            Magnetization = []
            

            for t in Time:
                value = object_vqe.magnetization( [], t, 7 )
                Magnetization.append( value )

        Result = pd.DataFrame( {'time': Time, 'values': Magnetization} )
        Result.to_csv("MagnetizationValues.csv")

    elif params["observable_params"]['observable'] == "Specific heat":
        sup = params["observable_params"]['lim_sup']
        inf = params["observable_params"]['lim_inf']
        n = params["observable_params"]['number_points']
        Temperature = np.linspace(inf, sup, n)

        object_vqt = spin_hamiltonian(params["hamiltonian_params"])
        object_vqt.set_device(params["ansatz_params"])
        object_vqt.set_node(params["ansatz_params"], [], 'Thermal')

        if object_vqt.correction != 0.5:
            raise Exception("Valor de espin no valido, considere solo espin 0.5")

        rep = params["ansatz_params"]["repetitions"]
        number = (object_vqt.qubits)*rep

        specific_heat = []
        for T in Temperature:
            theta = np.array( [np.random.randint(314)/100.0  for _ in range(number)] )
            dist = np.array( [np.random.randint(100)/100.0  for _ in range(object_vqt.qubits)] )

            if params["minimizate_method"] == "Gradient":
                theta = gradiend_method_VQT(object_vqt.thermal_cost_function, theta, dist, 1/T, params["minimizate_method_params"])
            elif params["minimizate_method"] == "Scipy":
                theta = scipy_method_VQT(object_vqt.thermal_cost_function, theta, dist, 1/T, params["minimizate_method_params"])
            else:
                raise Exception("Metodo de minimizacion ingresado no esta considerado")

            specific_heat.append( object_vqt.specific_heat(theta, T) )
        
        Result = pd.DataFrame( {'time': Temperature, 'values': specific_heat} )
        Result.to_csv("SpecificHeatValues.csv")

    else:
        raise Exception("Observable ingresado no encontrado")
    



elif params["simulation_object"] == "FermiHubbard":
    pass
else:
    raise Exception("Hamiltoniano ingreaso no valido")