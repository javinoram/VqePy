from pennylane import numpy as np
import pandas as pd
import pennylane as qml

def sum_even_odd_difference(arr):
    return np.abs( np.sum(arr[::2]) - np.sum(arr[1::2]) )

def pairs(arr):
    aux = [ -(np.abs(arr[2*i] + arr[(2*i)+1]) ) for i in range( int(len(arr)/2) ) ]
    return aux

def sort_states(states):
    return sorted(states, key=sum_even_odd_difference)

def map_characters(string):
    mapping = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    mapped_array = np.array([mapping[char] for char in string])
    return mapped_array

def group_string(l):
    mapping = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    mapped_list = np.array([map_characters(string) for string in l])
    div = np.array( [np.count_nonzero(term) if np.count_nonzero(term) else 1 for term in mapped_list.T])
    result = np.sum(mapped_list, axis=0)
    return [mapping[int(i)] for i in (result/div)]

def find_different_indices(input_string, character):
    char_array = np.array(list(input_string))
    return list(np.where(char_array != character)[0])

def list_to_string(l):
    string = ""
    for s in l:
        string+= s
    return string

def binary_to_string(state):
    return ''.join([str(i) for i in state])

def Pauli_function(term, qubits):
    auxiliar_string = ['I'] * qubits
    
    if isinstance(term, qml.ops.PauliZ):
        auxiliar_string[term.wires[0]] = 'Z'
    elif isinstance(term, qml.ops.PauliX):
        auxiliar_string[term.wires[0]] = 'X'
    elif isinstance(term, qml.ops.PauliY):
        auxiliar_string[term.wires[0]] = 'Y'
    elif isinstance(term, qml.ops.identity.Identity):
        pass
    elif isinstance(term, list):
        for pauli in term:
            if isinstance(pauli, qml.ops.PauliZ):
                auxiliar_string[pauli.wires[0]] = 'Z'
            elif isinstance(pauli, qml.ops.PauliX):
                auxiliar_string[pauli.wires[0]] = 'X'
            elif isinstance(pauli, qml.ops.PauliY):
                auxiliar_string[pauli.wires[0]] = 'Y'
    else:
        for pauli in term.non_identity_obs:
            if isinstance(pauli, qml.ops.PauliZ):
                auxiliar_string[pauli.wires[0]] = 'Z'
            elif isinstance(pauli, qml.ops.PauliX):
                auxiliar_string[pauli.wires[0]] = 'X'
            elif isinstance(pauli, qml.ops.PauliY):
                auxiliar_string[pauli.wires[0]] = 'Y'
    
    return ''.join(auxiliar_string)


def search_penalty_term(input_string, char):
    return any(input_string[i:i+2] == char for i in range(len(input_string) - 1))


def parity(integer, spin, qubits):
    binary = bin(integer)[2:].zfill(qubits)
    if spin == 0.5:
        binary = (binary.count("1"))%2
        if binary == 1:
            return -1
        else:
            return 1
    elif spin == 1:
        if search_penalty_term(binary, "11") or search_penalty_term(binary, "01"):
            return 0
        else:
            binary = (binary.count("1"))%2
            if binary == 1:
                return -1
            else:
                return 1
    elif spin == 1.5:
        return 0
    
    
def is_identity(term):
    if term.count("I") == len(term):
        return True
    else: 
        return False


def finite_diff(f, x, delta=0.01):
        gradient = []
        for i in range(len(x)):
            shift = np.zeros_like(x)
            shift[i] += 0.5 * delta
            res = (f(x + shift) - f(x - shift)) * delta**-1
            gradient.append(res)
        return gradient


def calculate_entropy(distribution):
    total_entropy = 0
    for d in distribution:
        total_entropy += -1 * d[0] * np.log(d[0]) + -1 * d[1] * np.log(d[1])
    return total_entropy


def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)


def prob_dist(params):
    return np.vstack([sigmoid(params), 1 - sigmoid(params)]).T



bohr_angs = 0.529177210903

conts_spin = { "1": {'0000':1, '1010':1, '0010': -1, '1000':-1},
            "1.5": {'0000':9/4, '0001':3/4, '0010':-3/4, '0011':-9/4, 
                '0100':3/4, '0101':1/4, '0110':-1/4, '0111':-3/4,
                '1000':-3/4, '1001':-1/4, '1010':1/4, '1011':-3/4,
                '1100':-9/4, '1101':-3/4, '1110':3/4, '1111':9/4},

            "2": {"1": {'000':2, '001':1, '010':0, '011':-1, '100':-2},
                      },
            "2.5": {"1": {'000':5/2, '001':3/2, '010':1/2, '011':-1/2, '100':-3/2, '101':-5/2},
                      },
            }