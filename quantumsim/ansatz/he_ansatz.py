import pennylane as qml
from .base import base_ansatz

"""
Clase de ansatz hardware efficient, esta construido para ser usado en sistemas de espines, 
hereda metodos de la clase base_ansatz.

Este ansatz esta hardcodeado para utilizar la variante de ansatz real amplitudes.
"""
class he_ansatz(base_ansatz):
    """
    Variables extras de la clase
    """
    pattern = "chain"
    begin_state = None


    """
    Funcion para setear el estado inicial del circuito
    input:
        state: vector de tamaño 2**n que representa un estado cuantico inicial para el circuito
    """
    def set_state(self, state):
        self.begin_state = state


    """
    Funcion para agregar compuertas de rotacion al circuito
    input:
        params: lista de valores de parametros de las compuertas de reotacion
    """
    def single_rotation(self, params):
        for i in range(0, self.qubits):
            qml.RY(params[i], wires=[i])


    """
    Funcion para agregar compuertas no locales al circuito (compuertas que agregan correlaciones 
    al circuito)
    input:
        flag: valor entero para determinar si se quieren colocar las compuertas en orden inverso o no
    """
    def non_local_gates(self):
        ## Compuertas en orden normal
        if self.pattern == 'chain':
            for i in range(0, self.qubits-1):
                qml.CNOT(wires=[i, i+1])

        elif self.pattern == 'ring':
            if self.qubits == 2:
                qml.CNOT(wires=[0, 1])
            else:
                for i in range(0, self.qubits-1):
                    qml.CNOT(wires=[i, i+1])
                qml.CNOT(wires=[self.qubits-1, 0])  

    def non_local_gates_inverse(self):
        #Compuertas en orden inverso (la operacion inversa)
        if self.pattern == 'chain':
            for i in range(self.qubits-1,0,-1):
                qml.CNOT(wires=[i-1, i])

        elif self.pattern == 'ring':
            if self.qubits == 2:
                qml.CNOT(wires=[0, 1])
            else:
                qml.CNOT(wires=[self.qubits-1, 0])      
                for i in range(self.qubits-1,0,-1):
                    qml.CNOT(wires=[i-1, i])

    


    """
    Circuito pararametrizado con el ansatz
    input:
        theta: vector de parametros del circuito
        obs: observable (en representacion de pennylane)
    output:
        valor: valor esperado del observable obs ingresado
    """
    def circuit(self, theta, obs):
        qml.StatePrep(self.begin_state, wires=range(self.qubits))
        rotation_number = self.qubits
        for k in range(0, self.repetition):
            params = theta[k*rotation_number:(k+1)*rotation_number]
            self.single_rotation(params)
            self.non_local_gates()
            
        return qml.expval(obs)


    """
    Circuito para obtener el estado al final del circuito
    input:
        theta: vector de parametros del circuito
    output:
        estado: retorna el estado del circuito como un arreglo numpy
    """
    def circuit_state(self, theta):
        qml.StatePrep(self.begin_state, wires=range(self.qubits))
        rotation_number = self.qubits
        for k in range(0, self.repetition):
            params = theta[k*rotation_number:(k+1)*rotation_number]
            self.single_rotation(params)
            self.non_local_gates()
        return qml.state()

