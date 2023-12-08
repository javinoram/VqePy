from .base import *

'''
Construtor del hamiltoniano de espines
'''
class vqe_spin(vqe_base):
    def __init__(self, params, lat):
        self.qubits = params['sites']
        terms = []
        coeff = []

        if lat['lattice'] == 'custom':
            pass
        else:
            x,y = lat['size']
            lattice_edge, lattice_node = lattice(lat)

            #J term
            for pair in lattice_edge:
                termz = qml.PauliZ(wires=[ x*pair[0][0] + pair[0][1]])@qml.PauliZ(wires=[x*pair[1][0] + pair[1][1]])
                termx = qml.PauliX(wires=[ x*pair[0][0] + pair[0][1]])@qml.PauliX(wires=[x*pair[1][0] + pair[1][1]])
                termy = qml.PauliY(wires=[ x*pair[0][0] + pair[0][1]])@qml.PauliY(wires=[x*pair[1][0] + pair[1][1]])

                terms.extend( [termx, termy, termz] )
                coeff.extend( [-params["Jx"][0]/4.0, -params["Jy"][1]/4.0, -params["Jz"][2]/4.0] )

            #h term
            if 'h' in params:
                for pair in lattice_node:
                    termz = qml.PauliZ(wires=[ x*pair[0] + pair[1]])
                    termz = qml.PauliX(wires=[ x*pair[0] + pair[1]])
                    termz = qml.PauliY(wires=[ x*pair[0] + pair[1]])

                    terms.extend( [termx, termy, termz] )
                    coeff.extend( [params["h"][0]/2.0, params["h"][1]/2.0, params["h"][2]/2.0] )
        
        to_delete = []
        for i,c in enumerate(coeff):
            if c==0.0:
                to_delete.append(i)
        
        to_delete = -np.sort(-np.array(to_delete))
        for index in to_delete:
            coeff.pop(index)
            terms.pop(index)

        self.hamiltonian = qml.Hamiltonian(np.real(np.array(coeff)), terms)
        return
    
