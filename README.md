# Project
Research project of using VQE for the study of condensed matter models

The main objectives of the proyect is the possibility to study to system:
1. Spin systems
2. 2D materials

For the first type, the idea is extend the methods for upper spins (>0.5).

# Libraries
1. qiskit
2. numpy
3. scipy
4. plotly
5. pandas
6. pennylane-qiskit

# Virtual enviroment
Create a virtual enviroment using:

´´´  python3 -m venv env ´´´

To activate the virtual enviroment, execute the following command in the main project's folder.

´´´ source env/bin/activate ´´´

It's recommended install other libraries inside the virtual enviroment. To install the requirement file use the following command

´´´ pip3 install -r requirements.txt ´´´


# Backends availables
The library Pennylane have a interation with qiskit, so the qiskit's backend are available to use in the pannylan's circuits. Here a list of the available qiskit's backends.
1. 'aer_simulator'
2. 'aer_simulator_statevector'
3. 'aer_simulator_density_matrix'
4. 'aer_simulator_stabilizer'
5. 'aer_simulator_matrix_product_state'
6. 'aer_simulator_extended_stabilizer'
7. 'aer_simulator_unitary', 
8. 'aer_simulator_superop'
9. 'qasm_simulator'
10. 'statevector_simulator'
11. 'unitary_simulator'
12. 'pulse_simulator'
