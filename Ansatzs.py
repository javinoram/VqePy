import numpy as np
import scipy.linalg as la
import math
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, BasicAer, execute

def variational_hamiltonian_ansatz(xs, qubits_number, hamiltonian_terms, index_list, reps):
    qc = QuantumCircuit( qubits_number, 2)
    for j in range(reps):
        for i, pauli_string in enumerate(hamiltonian_terms):
            if 'Y' in pauli_string[0]:
                qc.rx(np.pi/2, index_list[i][1:])
                qc.cnot( index_list[i][1], index_list[i][2])
                qc.rz(xs[i + len(hamiltonian_terms)*j], index_list[i][2])
                qc.cnot( index_list[i][1], index_list[i][2] )
                qc.rx(-np.pi/2, index_list[i][1:])
            
            if 'X' in pauli_string[0]:
                qc.ry(np.pi/2, index_list[i][1:])
                qc.cnot( index_list[i][1], index_list[i][2] )
                qc.rz(xs[i + len(hamiltonian_terms)*j], index_list[i][2])
                qc.cnot( index_list[i][1], index_list[i][2] )
                qc.ry(-np.pi/2, index_list[i][1:])
            
            if 'Z' in pauli_string[0]:
                qc.cnot( index_list[i][1], index_list[i][2] )
                qc.rz(xs[i + len(hamiltonian_terms)*j], index_list[i][2])
                qc.cnot( index_list[i][1], index_list[i][2] )
    return qc   

def hardware_efficient_ansatz( spin, xs, qubits_number, bits_number, reps):
    adjust_number = math.ceil((2*spin+1)/2.0)
    qc = QuantumCircuit( adjust_number*qubits_number, adjust_number*bits_number)

    if qubits_number == 1:
        for j in range(reps):
            for i in range(adjust_number):
                qc.ry( xs[j], i)
    else:
        for j in range(reps-1):
            '''
            Ry gates added to the circuit in block
            '''
            for i in range(qubits_number):
                for k in range(adjust_number):
                    qc.ry( xs[ i + qubits_number*j], adjust_number*i+k)
                    #qc.ry( xs[ (i + qubits_number*j)*adjust_number +k], adjust_number*i+k)

            '''
            CZ gates added to the circuit in block
            '''
            for qudit_index_1 in range(qubits_number-1):
                for qudit_index_2 in range(qudit_index_1+1, qubits_number):
                    for qubit_index_1 in range(adjust_number):
                        for qubit_index_2 in range(adjust_number):
                            #print(qudit_index_1, qudit_index_2, qubit_index_1, qubit_index_2)
                            #print(qudit_index_1*adjust_number+ qubit_index_1, qudit_index_2*adjust_number+ qubit_index_2)
                            qc.cz(qudit_index_1*adjust_number+ qubit_index_1, qudit_index_2*adjust_number+ qubit_index_2)
        for i in range(qubits_number):
            for k in range(adjust_number):
                qc.ry( xs[ i + qubits_number*(reps-1)], adjust_number*i+k)
                #qc.ry( xs[(i + qubits_number*(reps-1))*adjust_number +k], adjust_number*i+k)
                #qc.ry(xs[i + qubits_number*(reps-1)], qubits_number*i+k)
    return qc 


print(hardware_efficient_ansatz( 1, [i for i in range(100)], 1, 1, 3))