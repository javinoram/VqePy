import numpy as np
import scipy.linalg as la
import scipy as sc
import math
import itertools
import pennylane as qml
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, BasicAer, execute

def hardware_efficient_ansatz( spin, xs, qubits_number, bits_number, reps, state):
    adjust_number = math.ceil((2*spin+1)/2.0)
    qc = QuantumCircuit( adjust_number*qubits_number, adjust_number*bits_number)

    rotation_params = xs[(qubits_number)*reps:]
    controlled_params = xs[:(qubits_number)*reps]

    for i, s in enumerate(state):
        if s == 1:
            qc.x(i)

    if qubits_number == 1:
        for j in range(reps):
            for i in range(adjust_number):
                qc.ry( xs[j], i)
    else:
        for j in range(reps-1):
            '''
            Ry gates added to the circuit in block
            '''
            for i in range(qubits_number):
                for k in range(adjust_number):
                    qc.rz( rotation_params[0+ (i + qubits_number*j)*3], adjust_number*i+k)
                    qc.ry( rotation_params[1+ (i + qubits_number*j)*3], adjust_number*i+k)
                    qc.rx( rotation_params[2+ (i + qubits_number*j)*3], adjust_number*i+k)

            '''
            CZ gates added to the circuit in block
            '''
            for qubit_index in range(qubits_number):
                qc.crx(controlled_params[2*j + qubit_index], qubit_index, (qubit_index+1)%qubits_number)
            #for qudit_index_1 in range(qubits_number-1):
            #    for qudit_index_2 in range(qudit_index_1+1, qubits_number):
            #        for qubit_index_1 in range(adjust_number):
            #            for qubit_index_2 in range(adjust_number):
            #                qc.cz(qudit_index_1*adjust_number+ qubit_index_1, qudit_index_2*adjust_number+ qubit_index_2)
        for i in range(qubits_number):
            for k in range(adjust_number):
                qc.rz( rotation_params[0+ (i + qubits_number*(reps-1))*3], adjust_number*i+k)
                qc.ry( rotation_params[1+ (i + qubits_number*(reps-1))*3], adjust_number*i+k)
                qc.rx( rotation_params[2+ (i + qubits_number*(reps-1))*3], adjust_number*i+k)
    return qc 

def prob_dist(params):
    return np.vstack([sigmoid(params), 1 - sigmoid(params)]).T

def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)

def calculate_entropy(distribution):
    total_entropy = 0
    for d in distribution:
        total_entropy += -1 * d[0] * np.log(d[0]) + -1 * d[1] * np.log(d[1])
    return total_entropy

def single_rotation(phi_params, qubits):
    rotations = ["Z", "Y", "X"]
    for i in range(0, len(rotations)):
        qml.AngleEmbedding(phi_params[i], wires=qubits, rotation=rotations[i])

        

conts_spin = {"0.5": {"1": { '0':1, '1':-1, },
                      "2": {'00':1, '11':1, '01': -1, '10':-1}
                      },
            "1": {"1": {'00':1, '01':0, '10':-1, '11':0},
                      "2": {'0000':1, '1010':1, '0010': -1, '1000':-1}
                      },
            "1.5": {"1": {'00':3/2, '01':1/2, '10':-1/2, '11':-3/2},
                    "2": {'0000':9/4, '0001':3/4, '0010':-3/4, '0011':-9/4, 
                          '0100':3/4, '0101':1/4, '0110':-1/4, '0111':-3/4,
                          '1000':-3/4, '1001':-1/4, '1010':1/4, '1011':-3/4,
                          '1100':-9/4, '1101':-3/4, '1110':3/4, '1111':9/4}  },
            "2": {"1": {'000':2, '001':1, '010':0, '011':-1, '100':2},
                      },
            "2.5": {"1": {'000':5/2, '001':3/2, '010':1/2, '011':-1/2, '100':-3/2, '101':-5/2},
                      },
            }
