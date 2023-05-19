import pennylane as qml
from qiskit_ibm_provider import IBMProvider
token=''
IBMProvider.save_account(token= token, overwrite=True)

def circuit():
    qml.SingleExcitation(10, wires=[0,1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


provider = IBMProvider()

print("1")
#'ibm_lagos'
#'ibmq_qasm_simulator'
dev = qml.device('qiskit.ibmq', wires=2, backend='ibm_lagos', ibmqx_token=token)
print("2")
node = qml.QNode(circuit, dev, interface="autograd")
print("3")
result = node()
print(result)