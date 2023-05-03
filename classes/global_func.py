import math
import itertools
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc


def prob_dist(params):
    return np.vstack([sigmoid(params), 1 - sigmoid(params)]).T

def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)

def calculate_entropy(distribution):
    total_entropy = 0
    for d in distribution:
        total_entropy += -1 * d[0] * np.log(d[0]) + -1 * d[1] * np.log(d[1])
    return total_entropy
    
def number_rotation_params(rotation_set, qubits, reps):
        return len(rotation_set)*qubits*reps

def number_nonlocal_params(text, qubits, reps):
    if text=='chain':
        return int( (qubits-1)*reps )
    elif text=='ring':
        return int( qubits*reps )
    elif text=='all_to_all':
        return int( (qubits/2)*(qubits-1)*reps )
    else:
        return 0

bohr_angs = 0.529177210903

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