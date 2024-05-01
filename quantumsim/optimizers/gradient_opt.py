import pennylane as qml
from pennylane import numpy as np

"""
Clase del optimizador de gradiente (gradientes implementados en la libreria pennylane), 
el objetivo es calcular algunos flujos como el VQE o la optimizacion de estructuras.
"""
class gradiend_optimizer():
    """
    Variables de la clase
    """
    maxiter = 100
    theta_optimizer = None
    x_optimizer = None
    tensor_metric = None
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
            if params['theta'][0] == "generic":
                self.theta_optimizer = qml.GradientDescentOptimizer(stepsize=params['theta'][1])
            elif params['theta'][0] == "adam":
                self.theta_optimizer = qml.AdamOptimizer(stepsize=params['theta'][1])
            elif params['theta'][0] == "adagrad":
                self.theta_optimizer = qml.AdagradOptimizer(stepsize=params['theta'][1])

        if 'x' in params:
            if params['x'][0] == "generic":
                self.x_optimizer = qml.GradientDescentOptimizer(stepsize=params['x'][1])
            elif params['x'][0] == "adam":
                self.x_optimizer = qml.AdamOptimizer(stepsize=params['x'][1])
            elif params['x'][0] == "adagrad":
                self.x_optimizer = qml.AdagradOptimizer(stepsize=params['x'][1])
        
        self.begin_state = np.random.random( size=self.number )*(np.pi/180.0)


    """
    Funcion que ejecuta el flujo del VQE
    input: 
        cost_function: funcion de coste para el optimizador.
    output:
        energy_evol: lista con los valores de energia en cada iteracion.
        theta_evol: lista con los parametros del circuito en cada iteracion.
    """
    def VQE(self, cost_function):
        theta = self.begin_state
        energy_evol = [cost_function(theta)]
        theta_evol = [theta]

        for n in range(self.maxiter):
            theta.requires_grad = True
            theta = self.theta_optimizer.step(cost_function, theta)

            energy_evol.append(cost_function(theta))
            theta_evol.append(theta)

            prev_energy = energy_evol[-2]
            conv = np.abs(energy_evol[-1] - prev_energy)
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
           

    """
    Funcion que ejecuta el flujo del VQE para la relajacion de distancias entre moleculas
    input: 
        cost_function: funcion de coste para el optimizador
        x: posiciones iniciales de los elementos de la molecula
        grad: funcion que corresponde al gradiente de las posiciones de los elementos
    output:
        energy_evol: lista con los valores de energia en cada iteracion.
        theta_evol: lista con los parametros del circuito y las posiciones de los elementos
             en cada iteracion.
    """
    def OS(self, cost_function, x, grad):
        theta = self.begin_state
        theta_evol = [ np.concatenate( (x, theta)) ]
        energy_evol = [ cost_function( theta, x ) ]

        for _ in range(self.maxiter):
            theta.requires_grad = True
            x.requires_grad = False
            theta, _ = self.theta_optimizer.step(cost_function, theta, x)

            x.requires_grad = True
            theta.requires_grad = False
            _, x = self.x_optimizer.step(cost_function, theta, x, grad_fn=grad)


            theta_evol.append( np.concatenate( (x, theta) ) )
            energy_evol.append( cost_function( theta,x ) )


            if np.max( grad(theta, x) ) <= self.tol:
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
    