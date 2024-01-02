from .base import *

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

        if lat['lattice'] == 'custom':
            pass
        else:
            x,y = lat['size']
            lattice_edge, lattice_node = lattice(lat)

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
                Upotential = params["U"]
                fermi_U = 0.0
                for node in lattice_node:
                    p1, p2 = node
                    fermi_U += FermiC(y*p1 + 2*p2)*FermiA(y*p1 + 2*p2)*FermiC(y*p1 + 2*p2+1)*FermiA(y*p1 + 2*p2+1)
                fermi_sentence += Upotential*fermi_U

            #Construir terminos asociados al campo electroico E
            if 'E' in params:
                Efield = params["E"]
                fermi_E = 0.0
                for node in lattice_node:
                    p1, p2 = node
                    fermi_E += FermiC(y*p1 + 2*p2)*FermiA(y*p1 + 2*p2)
                    fermi_E += FermiC(y*p1 + 2*p2+1)*FermiA(y*p1 + 2*p2+1)
                fermi_sentence += Efield*fermi_E

            #Construir terminos asociados al potencial V
            if 'V' in params:
                Vpotencial = params["V"]
                fermi_V = 0.0
                for pair in lattice_edge:
                    p1, p2 = pair
                    n_i = FermiC(2*(y*p1[0] + p1[1]))*FermiA(2*(y*p1[0] + p1[1])) + FermiC(2*(y*p1[0] + p1[1]) +1)*FermiA(2*(y*p1[0] + p1[1]) +1)
                    n_j = FermiC(2*(y*p2[0] + p2[1]))*FermiA(2*(y*p2[0] + p2[1])) + FermiC(2*(y*p2[0] + p2[1]) +1)*FermiA(2*(y*p2[0] + p2[1]) +1)
                    fermi_V += n_i*n_j
                fermi_sentence += Vpotencial*fermi_V
        

        #Transformar los terminos de segunda cuantizacion a espines
        coeff, terms = qml.jordan_wigner( fermi_sentence, ps=True ).hamiltonian().terms()

        #Eliminar terminos cuyos coefficientes son 0
        to_delete = []
        for i,c in enumerate(coeff):
            if c==0.0:
                to_delete.append(i)
            if isinstance(terms[i], qml.Identity):
                terms[i] = qml.Identity(wires=[0])
        
        to_delete = -np.sort(-np.array(to_delete))
        for index in to_delete:
            coeff.pop(index)
            terms.pop(index)

        #Almacenar hamiltoniano
        self.hamiltonian = qml.Hamiltonian(np.real(np.array(coeff)), terms)
        return