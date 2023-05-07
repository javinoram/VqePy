import pennylane as qml
from classes.global_func import *

'''
Ansatz compose of XYZ rotation and CRX as a non local interaction.
This should be used in spin system with first neighbour interaction
'''
class Spin_ansatz():
    repetition: int = 0
    ansatz_pattern: str = ""
    number_params: list = []
    rotation_set: list = []

    def __init__(self, params, qubits) -> None:
        self.repetition = params['repetition']
        self.ansatz_pattern = params['ansatz_pattern']
        self.rotation_set = params['rotation_set']
        self.number_params = [number_rotation_params(self.rotation_set, qubits, self.repetition), 
            number_nonlocal_params(self.ansatz_pattern, qubits, self.repetition)]
        return

    def single_rotation(self, params, qubits, correction):
        for i in range( 0, qubits):
            for j in range(correction):
                qml.RZ(params[i][0], wires=[correction*i+j])
                qml.RY(params[i][1], wires=[correction*i+j])
                qml.RX(params[i][2], wires=[correction*i+j])
        return

    def non_local_gates(self, params, qubits, correction):
        if self.ansatz_pattern == 'chain':
            for i in range(0, qubits-1):
                for j in range(correction):
                    qml.CRX(params[i], [correction*i+j, correction*(i+1)+j])
        if self.ansatz_pattern == 'ring':
            if qubits == 2:
                for j in range(correction):
                    qml.CRX(params[i], [j, correction+j])
            else:
                for i in range(0, qubits-1):
                    for j in range(correction):
                        qml.CRX(params[i], [correction*i+j, correction*(i+1)+j])
        return

    def spin_circuit(self, qubits, correction, params, wire, init_state=None):
        qml.BasisState(init_state, wires=range(correction*qubits))
        for i in range(0, self.repetition):
            self.single_rotation(params[0][i], qubits, correction)
            self.non_local_gates(params[1][i], qubits, correction)

        aux = []
        for w in wire:
            for i in range(correction):
                aux.append( correction*w + i)
        return qml.counts(wires=aux)


'''
Ansatz compose of given rotation gates.
This should be used in electronic hamiltonians
'''
class Given_ansatz():
    repetition: int = 0
    singles: list = []
    doubles: list = []
    number_params: list = []
    hf_state = None

    def __init__(self, params, qubits) -> None:
        self.repetition = params['repetition']
        self.singles, self.doubles = qml.qchem.excitations(params['electrons'], qubits)
        self.number_params = [len(self.singles)*self.repetition, len(self.doubles)*self.repetition]
        return
    
    def given_circuit(self, qubits, params, hamiltonian):
        qml.BasisState(self.hf_state, wires=range(qubits))
        for _ in range(0, self.repetition):
            for i, term in enumerate(self.singles):
                qml.SingleExcitation(params[0][i], wires=term)
            for i, term in enumerate(self.doubles):
                qml.DoubleExcitation(params[1][i], wires=term)
        return qml.expval(hamiltonian)