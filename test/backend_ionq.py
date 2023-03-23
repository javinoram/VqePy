import pennylane as qml
from pennylane_ionq import ops

def circuit():
    qml.SingleExcitation(10, wires=[0,1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


IONQ_API_KEY = ''
print("1")
dev = qml.device("ionq.qpu", wires=[0,1], shots=1000, api_key= IONQ_API_KEY)
print("2")
node = qml.QNode(circuit, dev, interface="autograd")
print("3")
result = node()
print(result)