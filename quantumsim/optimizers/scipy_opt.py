import pennylane as qml
import scipy as sc
import warnings
from pennylane import numpy as np
from quantumsim.optimizers import *

class TookTooManyIters(Warning):
    pass


"""
Clase del optimizador con optimizadore de la libreria scipy, 
el objetivo es calcular algunos flujos como el VQE o la optimizacion de estructuras.
"""
class scipy_optimizer():
    """
    Variables de la clase
    """
    maxiter = 100
    type_method = "SLSQP"
    tol = 1e-6
    nit = 0
    number = 0
    begin_state= None

    """
    Constructor de la clase
    input: 
        params: diccionario con los parametros del optimizador
    """
    def __init__(self, params):
        self.number = params["number"]

        if 'tol' in params:
            self.tol = params["tol"]
        
        if 'maxiter' in params:
            self.maxiter = params["maxiter"]
        
        if 'type' in params:
            self.type_method = params["type"]

        self.begin_state = np.random.random( size=self.number )*(np.pi/180.0)


    def callback(self, x):
        self.nit += 1
        if self.nit == self.maxiter:
            print("Maximo numero de iteraciones")
            warnings.warn("Terminating optimization: iteration limit reached", TookTooManyIters)


    """
    Funcion que ejecuta el flujo del VQE
    input: 
        cost_function: funcion de coste para el optimizador.
    output:
        energy_evol: lista con los valores de energia en cada iteracion.
        theta_evol: lista con los parametros del circuito en cada iteracion.
    """
    def VQE(self, cost_function):
        energy_evol = []
        theta_evol = []
        self.nit = 0

        def cost_aux(x): 
            result = cost_function(x)
            energy_evol.append(result)
            theta_evol.append(x)
            return result
        
        ops = {'maxiter': self.maxiter}
        theta = sc.optimize.minimize(cost_aux, self.begin_state, method=self.type_method,
                callback=self.callback, tol=self.tol, options=ops)['x']
        return energy_evol, theta_evol
    

    """
    Funcion que ejecuta el flujo del VQE para la relajacion de distancias entre moleculas
    input: 
        cost_function: funcion de coste para el optimizador
        x: posiciones iniciales de los elementos de la molecula
    output:
        energy_evol: lista con los valores de energia en cada iteracion.
        theta_evol: lista con los parametros del circuito y las posiciones de los elementos
             en cada iteracion.
    """
    def OS(self, cost_function, x):
        energy_evol = []
        theta_evol = []
        self.nit = 0

        def cost_aux(psi): 
            result = cost_function(psi[len(x):], psi[:len(x)])
            energy_evol.append(result)
            theta_evol.append(psi)
            return result
        
        ops = {'maxiter': self.maxiter}
        theta = sc.optimize.minimize(cost_aux, np.concatenate((x,self.begin_state), axis=0), method=self.type_method, 
                callback=self.callback, tol=self.tol, options=ops)['x']
        return energy_evol, theta_evol


