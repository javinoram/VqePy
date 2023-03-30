import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
from numpy import array
import scipy
from scipy.optimize import minimize
import networkx as nx
import seaborn
import itertools

np.random.seed(42)

def cost_execution(params):

    global iterations

    cost = exact_cost(params)

    if iterations % 50 == 0:
        print("Cost at Step {}: {}".format(iterations, cost))

    iterations += 1
    return cost

def exact_cost(params):

    global iterations

    print("new iter")
    # Transforms the parameter list
    parameters = convert_list(params)
    dist_params = parameters[0]
    ansatz_params = parameters[1]

    # Creates the probability distribution
    distribution = prob_dist(dist_params)
    print(distribution)
    # Generates a list of all computational basis states of our qubit system
    combos = itertools.product([0, 1], repeat=nr_qubits)
    s = [list(c) for c in combos]
    print(s)
    # Passes each basis state through the variational circuit and multiplies
    # the calculated energy EV with the associated probability from the distribution
    cost = 0
    for i in s:
        result = qnode(ansatz_params[0], ansatz_params[1], sample=i)
        print(result)
        for j in range(0, len(i)):
            result = result * distribution[j][i[j]]
        cost += result

    # Calculates the entropy and the final cost function
    entropy = calculate_entropy(distribution)
    final_cost = beta * cost - entropy

    return final_cost

def convert_list(params):

    # Separates the list of parameters
    dist_params = params[0:nr_qubits]
    ansatz_params_1 = params[nr_qubits : ((depth + 1) * nr_qubits)]
    ansatz_params_2 = params[((depth + 1) * nr_qubits) :]

    coupling = np.split(ansatz_params_1, depth)

    # Partitions the parameters into multiple lists
    split = np.split(ansatz_params_2, depth)
    rotation = []
    for s in split:
        rotation.append(np.split(s, 3))

    ansatz_params = [rotation, coupling]

    return [dist_params, ansatz_params]

def calculate_entropy(distribution):

    total_entropy = 0
    for d in distribution:
        total_entropy += -1 * d[0] * np.log(d[0]) + -1 * d[1] * np.log(d[1])

    # Returns an array of the entropy values of the different initial density matrices

    return total_entropy

def quantum_circuit(rotation_params, coupling_params, sample=None):

    # Prepares the initial basis state corresponding to the sample
    qml.BasisStatePreparation(sample, wires=range(nr_qubits))

    # Prepares the variational ansatz for the circuit
    for i in range(0, depth):
        single_rotation(rotation_params[i], range(nr_qubits))
        qml.broadcast(
            unitary=qml.CRX,
            pattern="ring",
            wires=range(nr_qubits),
            parameters=coupling_params[i]
        )

    # Calculates the expectation value of the Hamiltonian with respect to the prepared states
    return qml.expval(qml.Hermitian(ham_matrix, wires=range(nr_qubits)))


def single_rotation(phi_params, qubits):

    rotations = ["Z", "Y", "X"]
    for i in range(0, len(rotations)):
        qml.AngleEmbedding(phi_params[i], wires=qubits, rotation=rotations[i])

def prob_dist(params):
    return np.vstack([sigmoid(params), 1 - sigmoid(params)]).T

def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)

def create_hamiltonian_matrix(n, graph):

    matrix = np.zeros((2 ** n, 2 ** n))

    for i in graph.edges:
        x = y = z = 1
        for j in range(0, n):
            if j == i[0] or j == i[1]:
                x = np.kron(x, qml.matrix(qml.PauliX)(0))
                y = np.kron(y, qml.matrix(qml.PauliY)(0))
                z = np.kron(z, qml.matrix(qml.PauliZ)(0))
            else:
                x = np.kron(x, np.identity(2))
                y = np.kron(y, np.identity(2))
                z = np.kron(z, np.identity(2))

        matrix = np.add(matrix, np.add(x, np.add(y, z)))

    return matrix

interaction_graph = nx.cycle_graph(3)
ham_matrix = create_hamiltonian_matrix(3, interaction_graph)
beta = 2  # beta = 1/T
nr_qubits = 3

depth = 4
dev = qml.device("default.qubit", wires=nr_qubits)
qnode = qml.QNode(quantum_circuit, dev, interface="autograd")
rotation_params = [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]] for i in range(0, depth)]
coupling_params = [[1, 1, 1, 1] for i in range(0, depth)]

iterations = 0

number = nr_qubits * (1 + depth * 4)
params = [np.random.randint(-300, 300) / 100 for i in range(0, number)]
out = minimize(cost_execution, x0=params, method="COBYLA", options={"maxiter": 500})
out_params = out["x"]

def prepare_state(params, device):

    # Initializes the density matrix

    final_density_matrix = np.zeros((2 ** nr_qubits, 2 ** nr_qubits))

    # Prepares the optimal parameters, creates the distribution and the bitstrings
    parameters = convert_list(params)
    dist_params = parameters[0]
    unitary_params = parameters[1]

    distribution = prob_dist(dist_params)

    combos = itertools.product([0, 1], repeat=nr_qubits)
    s = [list(c) for c in combos]

    # Runs the circuit in the case of the optimal parameters, for each bitstring,
    # and adds the result to the final density matrix

    for i in s:
        qnode(unitary_params[0], unitary_params[1], sample=i)
        state = device.state
        for j in range(0, len(i)):
            state = np.sqrt(distribution[j][i[j]]) * state
        final_density_matrix = np.add(final_density_matrix, np.outer(state, np.conj(state)))

    return final_density_matrix

# Prepares the density matrix
prep_density_matrix = prepare_state(out_params, dev)
seaborn.heatmap(abs(prep_density_matrix))
plt.show()