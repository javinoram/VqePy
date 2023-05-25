from pennylane import numpy as np
import pandas as pd

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
            "2": {"1": {'000':2, '001':1, '010':0, '011':-1, '100':2},
                      },
            "2.5": {"1": {'000':5/2, '001':3/2, '010':1/2, '011':-1/2, '100':-3/2, '101':-5/2},
                      },
            }