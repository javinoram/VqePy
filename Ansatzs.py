import numpy as np
import scipy.linalg as la
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

def hardware_efficient_ansatz(xs, qubits_number, bits_number, reps):
    qc = QuantumCircuit( qubits_number, bits_number)
    if qubits_number ==1:
        for j in range(reps):
            qc.ry( xs[j], 0)
    else:
        for j in range(reps-1):
            for i in range(qubits_number):
                qc.ry( xs[i + qubits_number*j], i)
            for i in range(qubits_number-1):
                for k in range(i,qubits_number-1):
                    qc.cz(i,k+1)
        for i in range(qubits_number):
            qc.ry(xs[i + qubits_number*(reps-1)], i)
    return qc 

def hardware_efficient_ansatz_overlap(xs, xs_prev, qubits_number, bits_number, reps):
    qc = QuantumCircuit( qubits_number, bits_number)
    if qubits_number ==1:
        for j in range(reps):
            qc.ry( xs_prev[j], 0)
    else:
        for i in range(qubits_number):
            qc.ry(xs_prev[i + qubits_number*(reps-1)], i)
        for j in range(reps-1):
            for i in range(qubits_number-1):
                for k in range(i,qubits_number-1):
                    qc.cz(i,k+1)
            for i in range(qubits_number):
                qc.ry( xs_prev[i + qubits_number*j], i)

    if qubits_number ==1:
        for j in range(reps):
            qc.ry( xs[j], 0)
    else:
        for j in range(reps-1):
            for i in range(qubits_number):
                qc.ry( xs[i + qubits_number*j], i)
            for i in range(qubits_number-1):
                for k in range(i,qubits_number-1):
                    qc.cz(i,k+1)
        for i in range(qubits_number):
            qc.ry(xs[i + qubits_number*(reps-1)], i)
    return qc 