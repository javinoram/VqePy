# VqePy

## Installation

VqePy requires Python 3.9 or newer.

To install VqePy on macOS or Linux, open a terminal and run:
```Shell
python3 -m pip install vqepy
```

Other way to do it, is clone the repository and and call all the functions adding the PATH.

## Tesis project
This library was a project to get a degree in computer science in the UTFSM University.

The main idea is implement a few variational quantum algorithms to study condensed matter and chemistry models and give a basic template to study complex phenomena. This project didnt want to be another quantum library. This should be see as a high level implementation to study systems with a fixed route to be executed, so, the user only need to give the parameters and just the minimun programming is needed.


## Parameters
Here, all the parameters of each class are listed.

### Hamiltonians

#### Molecules
The parameters considered here, come from the definition of the methods defined in pennylane [Molecular](https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.molecular_hamiltonian.html#pennylane.qchem.molecular_hamiltonian). Here we list the ones considered:
1. Mapping (mapping).
2. Charge (charge).
3. Multiplicity (mult).
4. Basis set (basis).
5. Differential method (method).
6. Active electrons (active_electrons).
7. Active orbitals (active_orbitals).

We recomended to not let any of this as a None value, to have a correct execution.

#### Fermi-Hubbard
The parameters considered here, come from the definition of the hamiltonian indicated here [Tenpy Hamiltonian](https://tenpy.readthedocs.io/en/latest/reference/tenpy.models.hubbard.FermiHubbardModel.html#tenpy.models.hubbard.FermiHubbardModel). Here we list the ones considerated:
1. t hopping (hopping).
2. On-site potencial U (U).
3. Potencial V (V).
4. Electric field (E).
5. Number of sites (sites).

We recomended to not let any of this as a None value, to have a correct execution.


#### Fermi-Hubbard
The parameters considered here, come from the definition of the hamiltonian indicated here [Wikipedia Hamiltonian](https://en.wikipedia.org/wiki/Quantum_Heisenberg_model). Here we list the ones considerated:
1. Exchange's vector (J): [Jx, Jy, Jz] .
2. Magnetic field's vector (h): [hx, hy, hz] .
3. Number of spins (sites).

The values of the exchange and magnetic field are the same for all the corresponding terms. We recomended to not let any of this as a None value, to have a correct execution.

### Ansatz
Here there are a lot of element of the base library to take into account, i recommended review:
1. [Base](https://docs.pennylane.ai/en/stable/code/qml_devices.html)
2. [Interface](https://docs.pennylane.ai/en/stable/code/qml_interfaces.html)
3. [Differential method](https://docs.pennylane.ai/en/stable/introduction/interfaces.html)

#### UCCSD Ansatz
This ansatz should be use with molecules and Fermi-Hubbard hamiltonian
1. Number of repetitions of the curcuit (repetitions): Here the number should 1
2. Base structure for circuits execution(base)
3. Interface of execution of the circuits (interface)
4. Number of electrons of the system (electrons)
5. Number of qubits of the circuit (qubits).
6. Differential method of the ansatz (diff_method): adjoint and best is recommended.

#### k-UpCCGSD Ansatz
This ansatz should be use with molecules and Fermi-Hubbard hamiltonian
1. Number of repetitions of the curcuit (repetitions)
2. Base structure for circuits execution(base)
3. Interface of execution of the circuits (interface)
4. Number of electrons of the system (electrons)
5. Number of qubits of the circuit (qubits).
6. Differential method of the ansatz (diff_method): adjoint and best is recommended.

#### Hardware Efficient Ansatz
This ansatz should be use with spin system
1. Number of repetitions of the curcuit (repetitions)
2. Base structure for circuits execution(base)
3. Interface of execution of the circuits (interface)
4. Number of electrons of the system (electrons)
5. Number of qubits of the circuit (qubits).
6. Differential method of the ansatz (diff_method): adjoint and best is recommended.
7. Pattern of the non local gates (pattert): chain or ring

There also the begin state of size 2**n that is needed.

#### Custom Ansatz
This ansatz should be use with molecules and Fermi-Hubbard hamiltonian
1. Number of repetitions of the curcuit (repetitions)
2. Base structure for circuits execution(base)
3. Interface of execution of the circuits (interface)
4. Number of electrons of the system (electrons)
5. Number of qubits of the circuit (qubits).
6. Differential method of the ansatz (diff_method): adjoint and best is recommended.

Other elements are the set of singles and doubled given gates obteined from the ADAPT-VQE. There also the begin state of size 2**n.

### Optimizers
#### Pennylane's optimizers
1. Number of parameters (number).
2. Maximum number of iterations (maxiter).
3. Theta optimizer (theta): list of optimizer (generic, adam or adagrad) and learning rate.
4. X optimizer (x): list of optimizer (generic, adam or adagrad) and learning rate.
5. Tolerance of relative error (tol).

#### Scipy's optimizers
1. Number of parameters (number).
2. Maximum number of iterations (maxiter).
3. Optimizer [scipy method section](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) (type).
4. Tolerance of relative error (tol).

#### Adapt optimizer
1. Maximum number of iterations (maxiter).
2. Optimizer (theta): list of optimizer (generic) and learning rate.
3. Tolerance of relative error (tol).
4. Number of electrons (electrons).
5. Number of qubits (qubits).
6. Sz proyection value (sz) (I recommend keek that value as 0).

#### Jax's optimizers
1. Number of parameters (number).
2. Maximum number of iterations (maxiter).
3. Theta optimizer (theta): list of optimizer (adam or adagrad) and learning rate.
4. X optimizer (x): list of optimizer (adam or adagrad) and learning rate.
5. Tolerance of relative error (tol).

### Lattice
For the construction of the lattice, we use the library networkx, for the purpose of the code, the parameters are the following:
1. Bound type (bound): boolean value, open (false) and periodic (close). 
2. Lattice type (lattice): chain, triangle, square, hexagon. 
3. Size of the lattice (size): tuple of integers (x,y).


I recommended reviewing my repository [my codes](https://github.com/javinoram/FH-Mol), where I have a few workflows using this code to compute VQE in molecules and 1D Fermi-Hubbard. This could be helpful to understand how to use it.
