from quantumsim.ansatz import *
from quantumsim.lattice import *
from pennylane import qchem
from pennylane import FermiC, FermiA
import itertools

"""
Clase base del VQE.
"""
class vqe_base():
    """
    Variables de la clase
    """
    hamiltonian= None
    coeff = None
    qubits = 0
    node = None
    interface = None


    """
    Funcion para setear el nodo de ejecucion del circuito
    input:
        nodo: nodo de ejecucion
        interface: interface en la cual se construyo el nodo
    """
    def set_node(self, node, interface) -> None:
        self.node = node
        self.interface = interface
        return
    

    """
    Funcion de coste.
    input:
        theta: vector de parametros del circuito
    output:
        result: evaluacion de la funcion de coste dado theta.
    """
    def cost_function(self, theta) -> float:
        result = self.node( theta=theta, obs=self.hamiltonian )
        return result
    

    """
    Funcion para obtener las proyeccion en la base computacional del vector de
    estado del circuito.
    input:
        theta: vector de parametros del circuito
    output:
        result: lista con el valor de las proyecciones
        s: lista con los estados de la base computacional
    """
    def get_projections(self, theta):
        combos = itertools.product([0, 1], repeat=self.qubits)
        s = [list(c) for c in combos]
        pro = [ qml.Projector(state, wires=range(self.qubits)) for state in s]
        result = [self.node( theta=theta, obs=state) for state in pro]
        return result, s
    
    
    """
    Funcion para calcular el espin total del estado del circuito
    input:
        theta: vector de parametros del circuito
        electrons: cantidad de electrones del sistema
    output:
        result: valor del espin total del estado del circuito
    """
    def get_totalspinS(self, theta, electrons):
        s_square = qml.qchem.spin2(electrons, self.qubits)
        result = self.node( theta=theta, obs=s_square )
        return result
    

    """
    Funcion para calcular la proyeccion Sz del estado del circuito
    input:
        theta: vector de parametros del circuito
        electrons: cantidad de electrones del sistema
    output:
        result: valor de la proyeccion Sz del estado del circuito
    """
    def get_totalspinSz(self, theta):
        s_z = qml.qchem.spinz(self.qubits)
        result = self.node( theta=theta, obs=s_z )
        return result
    

    """
    Funcion para calcular el numero de particulas del estado del circuito
    input:
        theta: vector de parametros del circuito
    output:
        result: valor del numero de particulas del estado del circuito
    """
    def get_particlenumber(self, theta):
        n = qml.qchem.particle_number(self.qubits)
        result = self.node( theta=theta, obs=n )
        return result

