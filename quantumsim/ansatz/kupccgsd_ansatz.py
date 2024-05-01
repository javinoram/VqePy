import pennylane as qml
from .base import base_ansatz


"""
Clase de ansatz k-UpCCGSD, esta construido para ser usado en molecules y modelo de Fermi-Hubbard, 
hereda metodos de la clase base_ansatz.
"""
class kupccgsd_ansatz(base_ansatz):

    """
    Variables extras de la clase
    """
    sz = 0
    begin_state = None

    
    """
    Funcion para setear el estado FH inicial del circuito
    input:
        electrons: numero de electrones del sistema
        sz: projeccion del sz usando reglas de seleccion que determina que compuertas es usaran, 
            se recomienda dejar este valor en 0
    """
    def set_state(self, electrons, sz):
        self.begin_state = qml.qchem.hf_state(electrons=electrons, orbitals=self.qubits)
        self.sz = sz


    """
    Circuito pararametrizado con el ansatz
    input:
        theta: vector de parametros del circuito
        obs: observable (en representacion de pennylane)
    output:
        valor: valor esperado del observable obs ingresado
    """
    def circuit(self, theta, obs):
        theta = theta.reshape( qml.kUpCCGSD.shape(k=self.repetition, 
            n_wires=self.qubits, delta_sz=self.sz) )
        
        qml.kUpCCGSD(theta, wires=range(self.qubits),
            k=self.repetition, delta_sz=0, init_state=self.begin_state)
        return qml.expval(obs)
    

    """
    Circuito para obtener el estado al final del circuito
    input:
        theta: vector de parametros del circuito
    output:
        estado: retorna el estado del circuito como un arreglo numpy
    """
    def circuit_state(self, theta):
        theta = theta.reshape( qml.kUpCCGSD.shape(k=self.repetition, 
            n_wires=self.qubits, delta_sz=self.sz) )
        
        qml.kUpCCGSD(theta, wires=range(self.qubits),
            k=self.repetition, delta_sz=0, init_state=self.begin_state)
        return qml.state()