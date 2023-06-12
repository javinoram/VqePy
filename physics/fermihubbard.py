from physics.functions.ansatz import *
from physics.functions.constans import *
from pennylane import qchem
import math
import itertools

'''
    1D Hubbard model, lineal
    Cada sitio es modelado usando dos qubits.
    Estos son ordenados agrupandolos segun el espin del sitio
    [sitios de espin up]_n [sitios de espin down]_n
'''
class variational_quantum_eigensolver_fermihubbard(given_fermihubbard_ansazt):
    hamiltonian_object= None
    hopping = 0.0
    potential = 0.0
    
    def __init__(self, params):
        self.qubits = params["sites"]*2
        self.hopping = params["hopping"]
        self.potential = params["potential"]

        fermi_sentence = {}
        hop = []
        pot = []
        coeff = []
        expression = []

        for i in range(params["sites"]-1):
            hop.append( qml.fermi.fermionic.FermiWord({(0, 2*i) : '+', (1, 2*i +2) : '-'}) )
            hop.append( qml.fermi.fermionic.FermiWord({(0, 2*i) : '-', (1, 2*i +2) : '+'}) )

            hop.append( qml.fermi.fermionic.FermiWord({(0, 2*i +1) : '+', (1, 2*i +3) : '-'}) )
            hop.append( qml.fermi.fermionic.FermiWord({(0, 2*i +1) : '-', (1, 2*i +3) : '+'}) )

            pot.append( qml.fermi.fermionic.FermiWord({(0, 2*i) : '+', (1, 2*i) : '-'}) )
            pot.append( qml.fermi.fermionic.FermiWord({(0, 2*i+1) : '+', (1, 2*i+1) : '-'}) )
            
        for term in hop:
            fermi_sentence[term] = -self.hopping
        for term in pot:
            fermi_sentence[term] = -self.potential


        aux = qml.fermi.FermiSentence(fermi_sentence)
        for term in aux:
            spin_space = qml.jordan_wigner( term )
            for op in spin_space:
                coeff_aux, operator_aux = op.terms()
                coeff.append( aux[term]*coeff_aux[0] )
                expression.append( operator_aux[0] )

        Pauli_terms = []
        for k, term in enumerate(expression):
            if type(term) not in [qml.ops.identity.Identity, qml.ops.qubit.non_parametric_ops.PauliZ,  qml.ops.qubit.non_parametric_ops.PauliX, qml.ops.qubit.non_parametric_ops.PauliY]:
                _, aux2 = term.terms()
                decomp_list = aux2[0].decomposition()
            string = Pauli_function(decomp_list, 6)
            Pauli_terms.append( [coeff[k], string] )

        self.hamiltonian_object = conmute_group(Pauli_terms)
    

    def density_charge(self, theta, time, n):
        number_pairs = int( self.qubits/2 )
        params = [theta[:self.repetition], theta[self.repetition:]]
        value_per_sites = []
        for i in range(number_pairs):
            result_down, result_up = self.node(theta = params, obs = [i, number_pairs + i], time= time, n=n, hamiltonian=self.hamiltonian_object)
            value_per_sites.append(result_up[1] + result_down[1])
        return value_per_sites
    

    def density_spin(self, theta, time, n):
        number_pairs = int( self.qubits/2 )
        params = [theta[:self.repetition], theta[self.repetition:]]
        value_per_sites = []
        for i in range(number_pairs):
            result_down, result_up = self.node(theta = params, obs = [i, number_pairs + i], time= time, n=n, hamiltonian=self.hamiltonian_object)
            value_per_sites.append(result_up[1] - result_down[1])
        return value_per_sites