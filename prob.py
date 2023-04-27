from ansatzs import *
from pennylane import qchem
from pennylane import numpy as np

Si = np.array([[1,0],[0,1]], dtype="float64") 
Sx = np.array([[0,1],[1,0]], dtype="float64")
Sy = np.array([[0,-1j],[1j,0]], dtype="complex64")
Sz = np.array([[1,0],[0,-1]], dtype="float64")

def single_rotation(phi_params, qubits):

    rotations = ["Z", "Y", "X"]
    for i in range(0, len(rotations)):
        qml.AngleEmbedding(phi_params[i], wires=qubits, rotation=rotations[i])

def quantum_circuit(rotation_params, coupling_params, sample=None):
    qml.BasisStatePreparation(sample, wires=range(nr_qubits))
    for i in range(0, depth):
        single_rotation(rotation_params[i], range(nr_qubits))
        qml.broadcast(
            unitary=qml.CRX,
            pattern="chain",
            wires=range(nr_qubits),
            parameters=coupling_params[i]
        )
    return qml.expval(qml.Hermitian(ham_matrix, wires=range(nr_qubits)))

def quantum_circuit2(rotation_params, coupling_params, wire, sample=None):
    qml.BasisStatePreparation(sample, wires=range(nr_qubits))
    for i in range(0, depth):
        single_rotation(rotation_params[i], range(nr_qubits))
        qml.broadcast(
            unitary=qml.CRX,
            pattern="chain",
            wires=range(nr_qubits),
            parameters=coupling_params[i]
        )
    return qml.probs(wires=wire)

ham_matrix = np.kron(Sz,np.kron(Sz,Si)) + np.kron(Si,np.kron(Sz,Sz))
nr_qubits = 3
depth= 4
dev = qml.device("default.qubit", wires=nr_qubits)
qnode1 = qml.QNode(quantum_circuit, dev, interface="autograd")
qnode2 = qml.QNode(quantum_circuit2, dev, interface="autograd")

rotation_params = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]] for i in range(0, depth)]
coupling_params = [[1, 1] for i in range(0, depth)]

result = qnode1(rotation_params, coupling_params, [0,0,0])

result2 = qnode2(rotation_params, coupling_params, [0,1], [0,0,0])
result3 = qnode2(rotation_params, coupling_params, [1,2], [0,0,0])
final_val1 = result2[0]+result2[3]-result2[1]-result2[2]
final_val2 = result3[0]+result3[3]-result3[1]-result3[2]

print(result, final_val1 + final_val2)
print(np.abs(result - (final_val1 + final_val2)))