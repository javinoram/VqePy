import pennylane as qml
import jax
import jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)

from pennylane import numpy as np
from quantumsim.optimizers import *


"""
Clase del optimizador que utiliza los optimizadores asociados a la libreria JAX, 
el objetivo es calcular algunos flujos como el VQE o la optimizacion de estructuras.
"""
class jax_optimizer():
    """
    Variables de la clase
    """
    maxiter = 100
    theta_optimizer = None
    x_optimizer = None
    tol = 1e-6
    number = 0
    begin_state= None
    get_energy = False
    get_params = False

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
        
        if 'theta' in params:
            if params['theta'][0] == "adam":
                self.theta_optimizer = optax.adam(learning_rate=params['theta'][1])
            elif params['theta'][0] == "adagrad":
                self.theta_optimizer = optax.adagrad(learning_rate=params['theta'][1])

        if 'x' in params:
            if params['theta'][0] == "adam":
                self.x_optimizer = optax.adam(learning_rate=params['theta'][1])
            elif params['theta'][0] == "adagrad":
                self.x_optimizer = optax.adagrad(learning_rate=params['theta'][1])

        self.begin_state = np.random.random( size=self.number )*(np.pi/180.0)


    """
    Funcion que ejecuta el flujo del VQE
    input: 
        cost_fn: funcion de coste para el optimizador.
    output:
        energy_evol: lista con los valores de energia en cada iteracion.
        theta_evol: lista con los parametros del circuito en cada iteracion.
    """
    def VQE(self, cost_fn):
        theta = jnp.array( self.begin_state )
        opt_state = self.theta_optimizer.init(theta)
        
        energy_evol = [cost_fn(theta)]
        theta_evol = [theta]

        for n in range(self.maxiter):

            grads = jax.grad(cost_fn)(theta)
            updates, opt_state = self.theta_optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)

            energy_evol.append(cost_fn(theta))
            theta_evol.append(theta)

            conv = jnp.abs(energy_evol[-1] - energy_evol[-2])
            if conv <= self.tol:
                break
        
        if self.get_energy == True and self.get_params == True:
            return energy_evol, theta_evol
        else:
            if self.get_energy == True:
                return energy_evol, theta_evol[-1]
            elif self.get_params== True:
                return energy_evol[-1], theta_evol
            else:
                return energy_evol[-1], theta_evol[-1]


