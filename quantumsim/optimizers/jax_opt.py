import pennylane as qml
import jax
import jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)

from pennylane import numpy as np
from quantumsim.optimizers import *

class jax_optimizer():
    maxiter = 100
    theta_optimizer = None
    x_optimizer = None
    tensor_metric = None
    tol = 1e-6
    number = 0
    begin_state= None

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



    def VQE(self, cost_fn):
        theta = jnp.array(self.begin_state)
        opt_state = self.theta_optimizer.init(theta)
        
        energy = [cost_fn(theta)]
        angle = [theta]

        for n in range(self.maxiter):
            grads = jax.grad(cost_fn)(theta)
            updates, opt_state = self.theta_optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)

            energy.append(cost_fn(theta))
            angle.append(theta)

            conv = jnp.abs(energy[-1] - energy[-2])
            if n % 1 == 0:
                print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

            if conv <= self.tol:
                break
        return energy, theta
    

    def OS(self, cost_fn, x, grad):
        theta = self.begin_state
        opt_state_theta = self.theta_optimizer.init(theta)

        opt_state_x = self.x_optimizer.init(x)
        x_evol = [x]
        energy_evol = [cost_fn(theta, x)]

        for n in range(self.maxiter):
            grads = jax.grad(cost_fn)(theta, x)
            updates_theta, opt_state_theta = self.theta_optimizer.update(grads, opt_state_theta)
            theta = optax.apply_updates(theta, updates_theta)


            updates_x, opt_state_x = self.x_optimizer.update(grads, opt_state_x)
            x = optax.apply_updates(x, updates_x)

            x_evol.append(x)
            energy_evol.append(cost_fn( theta, x ))
            if jnp.max(grad(theta, x)) <= self.tol:
                break
        return x, x_evol, theta, energy_evol



