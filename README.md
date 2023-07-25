# Tesis project
Project to get a degree in computer science.

The main idea is use variational quantum algorithms the study condensed matter's models,
the fields of interest are the following.
1. Molecular hamiltonian
2. Spin hamiltonian
3. Fermi-Hubbard hamiltonian

This project didnt want to be another quantum library. This should be see as a high level
implementation to study systems with a fixed route to be executed, so, the user only need to give 
a parameter's file and just the minimun programming is needed. 

# Libraries
All the code is done using as the base the pennylane library.
1. numpy
2. scipy
3. pennylane
4. matplotlib
5. pennylane-qiskit
6. qiskit_ibm_provider
7. openfermionpyscf
8. pyyaml
9. dask

# Requirements
It's recommended create a virtual enviroment and install all the libraries indicated in the requiremets.txt. 

``` pip3 install -r requirements.txt ```

# Methods
Here the three variational methods considerer are:

1. Variational quantum eigensolver
2. Variational quantum deflation
3. Variational quantum thermalizer

More information in the repository wiki.
