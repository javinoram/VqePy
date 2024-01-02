import pyscf
from pyscf import gto, scf, ci, cc
import numpy as np
import pennylane as qml

"""
Razon entre unidades angstrom y unidades atomicas.
"""
AngtoAU = 1.8897259886 #a.u


"""
Funcion para eliminar caracteres especiales de un string
input:
    input_string: Cadena de caracteres (string).
output:
    result: cadena con solo caracteres alfanumericos.
"""
def remove_special_characters(input_string):
    result = ''.join(char for char in input_string if char.isalnum() or char.isspace())
    return result


"""
Funcion para construir un estado inicial basado en CISD
input:
    elements: lista de elementos.
    coordinates: arreglo con las posiciones de los elementos en el espacio para cada elemento.
    params: diccionario de parametros de la estructura molecular.
output:
    estado: arreglo de numpy del estado inicial calculado usando tecnicas clasicas
"""
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


"""
Funcion para construir un estado inicial basado en CCSD
input:
    elements: lista de elementos.
    coordinates: arreglo con las posiciones de los elementos en el espacio para cada elemento.
    params: diccionario de parametros de la estructura molecular.
output:
    estado: arreglo de numpy del estado inicial calculado usando tecnicas clasicas
"""
def CCSD_state(elements, coordinates, params):
    atom = []
    for i in range( len(elements) ):
        atom.append( [elements[i], (( coordinates[3*i], coordinates[3*i+1], coordinates[3*i+2] )) ] )
   
    mol = gto.M(atom=atom, charge=params["charge"], unit="B")
    myhf = scf.RHF(mol).run()
    mycc = cc.CCSD(myhf).run()

    wf_cisd = qml.qchem.import_state(mycc, tol=1e-6)
    return wf_cisd.real
