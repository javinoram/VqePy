import numpy as np
import scipy as sc
from ansatzs import *


def Magnetization(theta: list, qubit_list, number_qubits, number_ansatz_repetition, backend, shots) -> float:
    circuit = hardware_efficient_ansatz(theta, number_qubits, 1, number_ansatz_repetition)
    func_val = 0.0
    for qubit_index in qubit_list:
        circuit_copy = circuit
        circuit_copy.measure([qubit_index], [0])
                
        job = execute( circuit_copy, backend, shots=shots)
        result = job.result().get_counts()
        if '0' in result:
            func_val += result['0']/shots
        else:
            func_val -= result['1']/shots
    return



def SpecificHeat(result_dict: dict) -> float:
    return



def Entropy(result_dict: dict) -> float:
    return