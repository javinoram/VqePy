import pennylane as qml
from .base import base_ansatz

"""
Clase de ansatz UCCSD, esta construido para ser usado en molecules y modelo de Fermi-Hubbard, 
hereda metodos de la clase base_ansatz.
"""
class uccds_ansatz(base_ansatz):
    """
    Variables extras de la clase
    """
    begin_state = None
    singles = None
    doubles = None


    """
    Funcion para setear los indices de las compuertas given que componen el ansatz.
    input:
        electrons: numero de electrones del sistema
        sz: projeccion del sz usando reglas de seleccion que determina que compuertas es usaran, 
            se recomienda dejar este valor en 0
    """
    def set_exitations(self, electrons, sz):
        singles, doubles = qml.qchem.excitations(electrons, self.qubits, delta_sz=sz)
        self.singles, self.doubles = qml.qchem.excitations_to_wires(singles, doubles)
    

    """
    Funcion para setear el estado FH inicial del circuito
    input:
        electrons: numero de electrones del sistema
        sz: projeccion del sz usando reglas de seleccion que determina que compuertas es usaran, 
            se recomienda dejar este valor en 0
    """
    def set_state(self, electrons, sz):
        self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)


    """
    Circuito pararametrizado con el ansatz
    input:
        theta: vector de parametros del circuito
        obs: observable (en representacion de pennylane)
    output:
        valor: valor esperado del observable obs ingresado
    """
    def circuit(self, theta, obs):
        qml.UCCSD(theta, range(self.qubits), self.singles, self.doubles, self.begin_state)
        return qml.expval(obs)
    

    """
    Circuito para obtener el estado al final del circuito
    input:
        theta: vector de parametros del circuito
    output:
        estado: retorna el estado del circuito como un arreglo numpy
    """
    def circuit_state(self, theta):
        qml.UCCSD(theta, range(self.qubits), self.singles, self.doubles, self.begin_state)
        return qml.state()
