import pennylane as qml
from pennylane import numpy as np
from .base import *


"""
Clase de ansatz customizado para ser usado junto al ADAPT-VQE, hereda metodos de la clase base_ansatz
"""
class custom_ansatz(base_ansatz):
    """
    Variables extras de la clase
    """
    begin_state = None
    excitations = []
    
    """
    Funcion para setear el estado inicial del circuito
    input:
        state: vector de tamaño 2**n que representa un estado cuantico inicial para el circuito
    """
    def set_state(self, state):
        self.begin_state = state


    """
    Funcion para indicar los qubits conectados con compuertan given
    input:
        singles: Lista de listas de pares de enteros que representan los indices
            de los qubits conectados por compuertas de rotacion given
        doubles: Lista de listas de pares de cuartetos que representan los indices
            de los qubits conectados por compuertas doble de rotacion given
    """
    def set_gates(self, singles=None, doubles=None):
        self.excitations = []
        if doubles != None:
            for term in doubles:
                self.excitations.append(term)

        if singles != None:
            for term in singles:
                self.excitations.append(term)
        return
    

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

        for i, gate in enumerate(self.excitations):
            if len(gate) == 4:
                qml.DoubleExcitation(theta[i], wires=gate)
            elif len(gate) == 2:
                qml.SingleExcitation(theta[i], wires=gate)

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

        for i, gate in enumerate(self.excitations):
            if len(gate) == 4:
                qml.DoubleExcitation(theta[i], wires=gate)
            elif len(gate) == 2:
                qml.SingleExcitation(theta[i], wires=gate)

        return qml.state()
    