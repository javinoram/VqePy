import pennylane as qml
from pennylane import numpy as np
from quantumsim.variational.vqe import vqe_spin, vqe_fermihubbard, vqe_molecular

#Test the hamiltonian of the h2 molecule
def test_hamiltonian_molecule():
    #pennylane hamiltonian
    elements=["H", "H"]
    coord=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    H,q = qml.qchem.molecular_hamiltonian(elements, coord)
    coef, ter = H.terms()
    H = qml.Hamiltonian(coef, ter)
    #vqepy class

    Hvqe = vqe_molecular(elements, coord, {})
    coefvqe, tervqe = Hvqe.hamiltonian.terms()
    Hvqe = qml.Hamiltonian(coefvqe, tervqe)
    assert H.compare(Hvqe)


