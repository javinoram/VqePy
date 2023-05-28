from pennylane import numpy as np
import pandas as pd
import pennylane as qml

def binary_to_string(state):
    return ''.join([str(i) for i in state])

def Pauli_function(term, qubits):
    auxiliar_string =[]
    if type(term)==qml.ops.qubit.non_parametric_ops.PauliZ:
        index = term.wires[0]
        for j in range(qubits):
            if j== index:
                auxiliar_string.append("Z")
            else:
                auxiliar_string.append("I")
                
    elif type(term)==qml.ops.qubit.non_parametric_ops.PauliX:
        index = term.wires[0]
        for j in range(qubits):
            if j== index:
                auxiliar_string.append("X")
            else:
                auxiliar_string.append("I")
                
    elif type(term)==qml.ops.qubit.non_parametric_ops.PauliY:
        index = term.wires[0]
        for j in range(qubits):
            if j== index:
                auxiliar_string.append("Y")
            else:
                auxiliar_string.append("I")

    elif type(term)==qml.ops.identity.Identity:
        for j in range(qubits):
            auxiliar_string.append("I")


    elif type(term) == list:
        auxiliar_string = ["I" for _ in range(qubits)]
        for pauli in term:
            index = pauli.wires[0]
            if type(pauli)==qml.ops.qubit.non_parametric_ops.PauliZ:
                auxiliar_string[index] = "Z"
            elif type(pauli)==qml.ops.qubit.non_parametric_ops.PauliX:
                auxiliar_string[index] = "X"
            elif type(pauli)==qml.ops.qubit.non_parametric_ops.PauliY:
                auxiliar_string[index] = "Y"
            else:
                pass
                
    else:
        Nonidentical = term.non_identity_obs
        auxiliar_string = ["I" for _ in range(qubits)]
        for pauli in Nonidentical:
            index = pauli.wires[0]
            if type(pauli)==qml.ops.qubit.non_parametric_ops.PauliZ:
                auxiliar_string[index] = "Z"
            elif type(pauli)==qml.ops.qubit.non_parametric_ops.PauliX:
                auxiliar_string[index] = "X"
            elif type(pauli)==qml.ops.qubit.non_parametric_ops.PauliY:
                auxiliar_string[index] = "Y"
            else:
                pass
    string = ""
    for s in auxiliar_string:
        string+= s
    return string


def conmute_group(pauli):
    pauli_aux = ['1' for i in range(len(pauli))]
    group_final = []

    for i in range(0, len(pauli_aux)):

        if pauli_aux[i] != '':
            aux = [ pauli[i] ]
            pauli_aux[i] = ''

            for j in range(len(pauli_aux)):
                if pauli_aux[j] != '':

                    auxm2 = pauli[j]
                    if conmute(aux, auxm2) and auxm2 not in aux and same_lenght(aux, auxm2):
                        aux.append(auxm2)
                        pauli_aux[j] = ""

            group_final.append(aux)
    return group_final

def same_lenght(list_terms, op):
    aux = 0
    for i in op[1]:
        if i != 'I':
            aux+=1
    for term in list_terms:
        aux_term = 0
        for i in term[1]:
            if i != 'I':
                aux_term+=1
        if aux_term !=aux:
            return False
    return True


def conmute(listm, m):
    m2 = m[1]
    for term in listm:
        s = term[1]
        for i in range(len(s)):
            if s[i] == 'Z':
                if m2[i] == 'X' or m2[i] == 'Y':
                    return False
            elif s[i] == 'X':
                if m2[i] == 'Y' or m2[i] == 'Z':
                    return False                
            elif s[i] == 'Y':
                if m2[i] == 'X' or m2[i] == 'Z':
                    return False
    return True

def parity(integer):
    binary = format(integer, 'b')
    aux = 0
    for s in binary:
        if s=='1':
            aux +=1
    if aux%2 == 0:
        return 1
    else:
        return -1
    
def is_identity(term):
    for i in range(len(term)):
        if term[i]!='I':
            return False
    return True

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
            "2": {"1": {'000':2, '001':1, '010':0, '011':-1, '100':-2},
                      },
            "2.5": {"1": {'000':5/2, '001':3/2, '010':1/2, '011':-1/2, '100':-3/2, '101':-5/2},
                      },
            }