import pennylane as qml
from pennylane import numpy as np
import jax

"""
Clase padre de los ansatz, donde se definen las caracteristicas y funciones
que todo ansatz debe tener.
"""
class base_ansatz():
    base = "default.qubit"
    backend = ""
    token = ""
    interface = "autograd"
    diff_method = "best"

    device= None
    node = None
    qubits = 0
    repetition = 0

    """
    Funcion para setear caracteristicas del entorno de ejecucion del circuito
    input:
        params: diccionario de parametros para inicializar el entorno de ejecucion 
        del circuito
    """
    def set_device(self, params) -> None:
        self.base = params['base']
        self.qubits = params["qubits"]

        ##Maquinas reales
        if self.base == 'qiskit.ibmq':
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            if params['token']:
                self.token = params['token']
                self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits,  ibmqx_token= self.token)
            else:
                raise Exception("Token de acceso no encontrado")
        ##Simuladores de qiskit
        elif self.base == "qiskit.aer":
            if params['backend']:
                self.backend = params['backend']
            else:
                raise Exception("Backend no encontrado")
            
            self.device= qml.device(self.base, backend=self.backend, wires=self.qubits)
        ##Simuladores de pennylane
        else:
            self.device= qml.device(self.base, wires=self.qubits)
        return
    
    """
    Funcion para setear caracteristicas del ejecutor del circuito
    input:
        params: diccionario de parametros para inicializar el ejecutor del circuito
    """
    def set_node(self, params) -> None:
        if "pattern" in params:
            self.pattern = params["pattern"]
        
        if params['repetitions']:
            self.repetition = params['repetitions']
        
        if params['interface']:
            self.interface= params['interface']
        
        if params['diff_method']:
            self.diff_method = params['diff_method']
        
        if params['interface'] == "jax" or params['interface'] == "jax-jit":
            node = qml.QNode(self.circuit, self.device, interface=self.interface, diff_method=self.diff_method)
            self.node = jax.jit(node)
        else:
            self.node = qml.QNode(self.circuit, self.device, interface=self.interface, diff_method=self.diff_method)
        return
    

    """
    Funcion para obtener el estado al final del circuito, despues de aplicar el ansatz, esto 
    solo funciona con simuladores
    input:
        theta: vector de parametros del circuito
    output:
        estado: retorna el estado del circuito como un arreglo numpy
    """
    def get_state(self, theta):
        if self.interface == "jax" or self.interface == "jax-jit":
            node = qml.QNode(self.circuit_state, self.device, interface=self.interface, diff_method=self.diff_method)
            node = jax.jit(node)
        else:
            node = qml.QNode(self.circuit_state, self.device, interface=self.interface, diff_method=self.diff_method)
        return node(theta)