import pennylane as qml
from pennylane import numpy as np
from quantumsim.ansatz import *
from quantumsim.lattice import *
from pennylane import qchem
from pennylane import FermiC, FermiA

"""
Clase base del proceso ADAPT-VQE, en esta se almacenan las variables y funciones generales, que son 
independientes del sistema
"""
class adap_base():
    """
    Variables de la clase
    """
    hamiltonian = None
    qubits = 0
    base = ""
    backend = ""
    token = ""
    device= None
    node = None
    begin_state = None

    """
    Funcion para setear caracteristicas del entorno de ejecucion del circuito
    input:
        params: diccionario de parametros para inicializar el entorno de ejecucion 
        del circuito
    """
    def set_device(self, params) -> None:
        self.base = params['base']

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

            
        if params['interface'] == "jax" or params['interface'] == "jax-jit":
            node = qml.QNode(self.circuit, self.device, interface=params['interface'])
            self.node = jax.jit(node)
        else:
            self.node = qml.QNode(self.circuit, self.device, interface=params['interface'])
        return
    

    """
    Funcion para setear el estado FH inicial del circuito
    input:
        electrons: numero de electrones del sistema
    """
    def set_state(self, electrons):
        self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)


    """
    Circuito pararametrizado para el proceso de ADAPT-VQE
    input:
        params: vector de parametros del circuito
        excitations: lista de compuertas
        params_select: lista de los parametros de las compuertas dobles seleccionadas
        gates_select: lista de las compuertas dobles selecciondas
    output:
        valor: valor esperado del hamiltoniano
    """
    def circuit(self, params, excitations, params_select=None, gates_select=None):
        qml.BasisState(self.begin_state, wires=range(self.qubits))

        if gates_select != None:
            for i, gate in enumerate(gates_select):
                if len(gate) == 4:
                    qml.DoubleExcitation(params_select[i], wires=gate)
                elif len(gate) == 2:
                    qml.SingleExcitation(params_select[i], wires=gate)

        for i, gate in enumerate(excitations):
            if len(gate) == 4:
                qml.DoubleExcitation(params[i], wires=gate)
            elif len(gate) == 2:
                qml.SingleExcitation(params[i], wires=gate)
        return qml.expval(self.hamiltonian)

