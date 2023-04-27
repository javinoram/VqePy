#import numpy as np
from pennylane import numpy as np

import math
import itertools
import pennylane as qml


def prob_dist(params):
    return np.vstack([sigmoid(params), 1 - sigmoid(params)]).T

def sigmoid(x):
    #aux = []
    #for i in x:
    #    aux.append( math.exp(i)/ (math.exp(i) + 1) )
    #return np.array(aux)
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
