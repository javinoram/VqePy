from quantumsim.ansatz import *
from quantumsim.lattice import *
from pennylane import qchem
from pennylane import FermiC, FermiA
import itertools

'''
Base del VQE donde se definen las funciones de coste,
y funciones de calculos auxiliares, como las proyecciones
'''
class vqe_base():
    hamiltonian= None
    coeff = None
    qubits = 0
    node = None

    def process_group(self, theta, i):
        result = self.node( theta=theta, obs= self.hamiltonian[i] )
        result = np.array(self.coeff[i]) @ np.array(result)
        return result 
    
    def cost_function(self, theta):
        results = [ self.process_group(theta, i) for i in range(len(self.hamiltonian)) ]
        return np.sum( results )
    
    #Cantidades utiles de saber
    #Probabilidad de cada uno de los estados basales segun el estado obtenido
    def get_projections(self, theta):
        combos = itertools.product([0, 1], repeat=self.qubits)
        s = [list(c) for c in combos]
        pro = [ qml.Projector(state, wires=range(self.qubits)) for state in s]
        result = [self.node( theta=theta, obs=[state])[0] for state in pro]
        return result, s
    
    #Espin total del estado
    def get_totalspinS(self, theta, electrons):
        s_square = qml.qchem.spin2(electrons, self.qubits)
        result = self.node( theta=theta, obs=[s_square] )[0]
        return result
    
    #Espin total proyectado en S_z del estado
    def get_totalspinSz(self, theta, electrons):
        s_z = qml.qchem.spinz(electrons, self.qubits)
        result = self.node( theta=theta, obs=[s_z] )[0]
        return result
    
    #Numero de particulas del estado
    def get_totalspinSz(self, theta, electrons):
        n = qml.qchem.particle_number(self.qubits)
        result = self.node( theta=theta, obs=[n] )[0]
        return result

