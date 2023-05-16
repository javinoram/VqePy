import pennylane as qml
import math
import scipy.linalg as la
import scipy as sc


'''
Generic minimization functions in scipy
'''
def scipy_minimization(cost_function, theta, params):
    xs = sc.optimize.minimize(cost_function, theta, method=params["method"],)['x']
    result = cost_function(xs)
    return result, xs

'''
Gradient methods defined in pennylane
'''
def generic_gradiend_method():
    return
