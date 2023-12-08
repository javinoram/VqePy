import pyscf
from pyscf import gto, scf, ci, cc
import numpy as np
import pennylane as qml

AngtoAU = 1.8897259886 #a.u

def remove_special_characters(input_string):
    result = ''.join(char for char in input_string if char.isalnum() or char.isspace())
    return result

def CISD_state(elements, coordinates, params):
    atom = []
    for i in range( len(elements) ):
        atom.append( [elements[i], (( coordinates[3*i], coordinates[3*i+1], coordinates[3*i+2] )) ] )
   
    basis = remove_special_characters( params["basis"] )
    mol = gto.M(atom=atom, charge=params["charge"], basis=basis, unit="B")
    myhf = scf.RHF(mol).run()
    myci = ci.CISD(myhf).run()
    wf_cisd = qml.qchem.import_state(myci, tol=1e-6)
    return wf_cisd.real



def CCSD_state(elements, coordinates, params):
    atom = []
    for i in range( len(elements) ):
        atom.append( [elements[i], (( coordinates[3*i], coordinates[3*i+1], coordinates[3*i+2] )) ] )
   
    mol = gto.M(atom=atom, charge=params["charge"], unit="B")
    myhf = scf.RHF(mol).run()
    mycc = cc.CCSD(myhf).run()

    wf_cisd = qml.qchem.import_state(mycc, tol=1e-6)
    return wf_cisd.real


def HF_state():


    return
