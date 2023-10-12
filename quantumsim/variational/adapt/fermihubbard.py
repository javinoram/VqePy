from .base import *

class adap_fermihubbard(adap_base):

    def __init__(self, params, lat):
        self.qubits = params["sites"]*2
        fermi_sentence = 0.0

        if lat['lattice'] == 'custom':
            pass
        else:
            x,y = lat['size']
            lattice_edge, lattice_node = lattice(lat)

            #-t term
            fermi_hopping = 0.0
            hopping = -params["hopping"]
            for pair in lattice_edge:
                p1, p2 = pair
                fermi_hopping +=  FermiC(2*(y*p1[0] + p1[1]))*FermiA(2*(y*p2[0] + p2[1])) + FermiC(2*(y*p2[0] + p2[1]))*FermiA(2*(y*p1[0] + p1[1]))
                fermi_hopping +=  FermiC(2*(y*p1[0] + p1[1])+1)*FermiA(2*(y*p2[0] + p2[1])+1) + FermiC(2*(y*p2[0] + p2[1])+1)*FermiA(2*(y*p1[0] + p1[1])+1)
            fermi_sentence = hopping*fermi_hopping

            #U term
            if 'on-site' in params['interactions']:
                fermi_U = 0.0
                Upotential = params["potential"]
                for node in lattice_node:
                    p1, p2 = node
                    fermi_U += FermiC(y*p1 + 2*p2)*FermiA(y*p1 + 2*p2)*FermiC(y*p1 + 2*p2+1)*FermiA(y*p1 + 2*p2+1)
                fermi_sentence += Upotential*fermi_U

            #E term
            if 'electric' in params['interactions']:
                fermi_E = 0.0
                Efield = params["electric"]
                for node in lattice_node:
                    p1, p2 = node
                    fermi_E += FermiC(2*(y*p1[0] + p1[1]))*FermiA(2*(y*p1[0] + p1[1]))
                    fermi_E += FermiC(2*(y*p2[0] + p2[1])+1)*FermiA(2*(y*p2[0] + p2[1])+1)
                fermi_sentence += Efield*fermi_E
        
        #fermi_sentence = self.hopping*fermi_hopping + self.potential*fermi_U
        coeff, terms = qml.jordan_wigner( fermi_sentence, ps=True ).hamiltonian().terms()

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
        self.hamiltonian = qml.Hamiltonian(np.real(np.array(coeff)), terms)
        return