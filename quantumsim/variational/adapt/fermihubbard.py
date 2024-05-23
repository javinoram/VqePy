from .base import adap_base
import pennylane as qml
from quantumsim.lattice import lattice, custom_lattice
from pennylane import numpy as np
from pennylane import FermiC, FermiA

"""
Clase del modelo de Fermi Hubbard, esta se encarga de construir el hamiltoniano,
hereda metodos de la clase adap_base
"""
class adap_fermihubbard(adap_base):

    """
    Constructor de la clase
        params: diccionario con los parametros del hamiltoniano
        lat: diccionario con los parametros de la lattice
    """
    def __init__(self, params, lat):
        self.qubits = params["sites"]*2
        fermi_sentence = 0.0
        
        x,y = lat['size']
        if lat['lattice'] != 'custom':
            lattice_edge, lattice_node = lattice(lat)
        else:
            lattice_edge, lattice_node = custom_lattice(lat)

        #Construir terminos asociados al termino -t
        hopping = -params["hopping"]
        fermi_hopping = 0.0
        for pair in lattice_edge:
            p1, p2 = pair
            fermi_hopping +=  FermiC(2*(y*p1[0] + p1[1]))*FermiA(2*(y*p2[0] + p2[1])) + FermiC(2*(y*p2[0] + p2[1]))*FermiA(2*(y*p1[0] + p1[1]))
            fermi_hopping +=  FermiC(2*(y*p1[0] + p1[1])+1)*FermiA(2*(y*p2[0] + p2[1])+1) + FermiC(2*(y*p2[0] + p2[1])+1)*FermiA(2*(y*p1[0] + p1[1])+1)

        fermi_sentence = hopping*fermi_hopping

        #Construir terminos asociados al potencial U
        if 'U' in params:
            u_potential = params["U"]
            fermi_u = 0.0
            for node in lattice_node:
                p1, p2 = node
                fermi_u += FermiC(y*p1 + 2*p2)*FermiA(y*p1 + 2*p2)*FermiC(y*p1 + 2*p2+1)*FermiA(y*p1 + 2*p2+1)
            fermi_sentence += u_potential*fermi_u

        #Construir terminos asociados al campo electroico E
        if 'E' in params:
            e_field = params["E"]
            fermi_e = 0.0
            for node in lattice_node:
                p1, p2 = node
                fermi_e += FermiC(y*p1 + 2*p2)*FermiA(y*p1 + 2*p2)
                fermi_e += FermiC(y*p1 + 2*p2+1)*FermiA(y*p1 + 2*p2+1)
            fermi_sentence += e_field*fermi_e

        #Construir terminos asociados al potencial V
        if 'V' in params:
            v_potencial = params["V"]
            fermi_v = 0.0
            for pair in lattice_edge:
                p1, p2 = pair
                n_i = FermiC(2*(y*p1[0] + p1[1]))*FermiA(2*(y*p1[0] + p1[1])) + FermiC(2*(y*p1[0] + p1[1]) +1)*FermiA(2*(y*p1[0] + p1[1]) +1)
                n_j = FermiC(2*(y*p2[0] + p2[1]))*FermiA(2*(y*p2[0] + p2[1])) + FermiC(2*(y*p2[0] + p2[1]) +1)*FermiA(2*(y*p2[0] + p2[1]) +1)
                fermi_v += n_i*n_j
            fermi_sentence += v_potencial*fermi_v
        

        #Transformar los terminos de segunda cuantizacion a espines
        coeff, terms = qml.jordan_wigner( fermi_sentence, ps=True ).hamiltonian().terms()

        #Eliminar terminos cuyos coefficientes son 0
        to_delete = [i for i,c in enumerate(coeff) if np.abs(c)<1e-10 ]
        for i,_ in enumerate(coeff):
            if isinstance(terms[i], qml.Identity):
                terms[i] = qml.Identity(wires=[0])
        
        to_delete = -np.sort(-np.array(to_delete))
        for index in to_delete:
            coeff.pop(index)
            terms.pop(index)

        #Almacenar hamiltoniano
        self.hamiltonian = qml.Hamiltonian(np.real(np.array(coeff)), terms)
