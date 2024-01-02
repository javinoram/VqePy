from .base import *

"""
Clase para construir el hamiltoniano usado en el proceso de VQE, hereda metodos de la clase
vqe_base
"""
class vqe_spin(vqe_base):

    """
    Constructor de la clase
        params: diccionario con los parametros del hamiltoniano
        lat: diccionario con los parametros de la lattice
    """
    def __init__(self, params, lat):
        self.qubits = params['sites']
        terms = []
        coeff = []

        if lat['lattice'] == 'custom':
            pass
        else:
            x,y = lat['size']
            lattice_edge, lattice_node = lattice(lat)

            #Construir terminos asociados al exchange -J
            for pair in lattice_edge:
                termz = qml.PauliZ(wires=[ x*pair[0][0] + pair[0][1]])@qml.PauliZ(wires=[x*pair[1][0] + pair[1][1]])
                termx = qml.PauliX(wires=[ x*pair[0][0] + pair[0][1]])@qml.PauliX(wires=[x*pair[1][0] + pair[1][1]])
                termy = qml.PauliY(wires=[ x*pair[0][0] + pair[0][1]])@qml.PauliY(wires=[x*pair[1][0] + pair[1][1]])

                terms.extend( [termx, termy, termz] )
                coeff.extend( [-params["J"][0]/4.0, -params["J"][1]/4.0, -params["J"][2]/4.0] )

            #Construir terminos asociados al campo magnetico h
            if 'h' in params:
                for pair in lattice_node:
                    termz = qml.PauliZ(wires=[ x*pair[0] + pair[1]])
                    termz = qml.PauliX(wires=[ x*pair[0] + pair[1]])
                    termz = qml.PauliY(wires=[ x*pair[0] + pair[1]])

                    terms.extend( [termx, termy, termz] )
                    coeff.extend( [params["h"][0]/2.0, params["h"][1]/2.0, params["h"][2]/2.0] )
        
        #Eliminar terminos cuyos coefficientes son 0
        to_delete = []
        for i,c in enumerate(coeff):
            if c==0.0:
                to_delete.append(i)
        
        to_delete = -np.sort(-np.array(to_delete))
        for index in to_delete:
            coeff.pop(index)
            terms.pop(index)

        #Almacenar hamiltonianos
        self.hamiltonian = qml.Hamiltonian(np.real(np.array(coeff)), terms)
        return
    
