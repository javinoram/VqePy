from quantumsim.ansatz import *
from quantumsim.optimizers.funciones import *
from pennylane import qchem
from pennylane import FermiC, FermiA
import math
import dask
from autograd.numpy.numpy_boxes import ArrayBox

'''
Clase con las funciones de coste para utilizar VQE y VQD en 
un hamiltoniano molecular
'''
class vqe_molecular():
    hamiltonian_object= None
    groups_caractericts = None
    coeff_object = None
    parity_terms = None
    qubits = 0

    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'

    node = None
    node_overlap = None

    def __init__(self, symbols, coordinates, params= None):
        if 'mapping' in params:
            if params['mapping'] in ("jordan_wigner"):
                self.mapping = params['mapping']
            else:
                raise Exception("Mapping no valido, considere jordan_wigner")
            
        if 'charge' in params:
            self.charge = params['charge']

        if 'mult' in params:
            self.mult = params['mult']

        if 'basis' in params:
            if params['basis'] in ("sto-3g", "6-31g", "6-311g", "cc-pvdz"):
                self.basis = params['basis']
            else:
                raise Exception("Base no valida, considere sto-3g, 6-31g, 6-311g, cc-pvdz")
        
        if 'method' in params:
            if params['method'] in ("pyscf", "dhf"):
                self.method = params['method']
            else:
                raise Exception("Metodo no valido, considere dhf o pyscf")

        
        aux_h, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method)
        coeff, terms = aux_h.terms()
        del aux_h

        terms, coeff = qml.pauli.group_observables(observables=terms, coefficients=coeff, grouping_type='qwc', method='rlf')
        Pauli_terms = [] 
        for group in terms: 
            aux = []
            for term in group:
                string = Pauli_function(term, self.qubits)
                aux.append(string)
            Pauli_terms.append(aux)

        self.hamiltonian_object = Pauli_terms
        self.coeff_object = coeff
        self.parity_terms = np.array([ parity(i, 0.5, self.qubits) for i in range(2**self.qubits) ]) 
        return
    

    def set_group_characteristics(self):
        aux_char = []
        for group in self.hamiltonian_object:
            aux_char.append( group_string(group) )
        self.groups_caractericts = aux_char
        return


    def process_group(self, theta, i):
        expval = []
        term = self.hamiltonian_object[i]
        charac = self.groups_caractericts[i]

        result_probs = self.node(theta=theta, obs=term, characteristic=charac)
        for k, probs in enumerate(result_probs):
            if is_identity(term[k]):
                expval.append( np.sum( probs ) )
            else:
                expval.append( np.sum( probs * self.parity_terms[:probs.shape[0]] ) )

        result = self.coeff_object[i]*np.array(expval)
        return np.sum(result)
    

    def cost_function(self, theta):
        results = []
        for i in range(len(self.hamiltonian_object)):
            results.append( self.process_group(theta, i) )
        return np.sum( results )


    def cost_function_parallel(self, theta):
        results = []
        for i in range(len(self.hamiltonian_object)):
            results.append( dask.delayed(self.process_group)(theta, i) )
        num_workers = 4
        result = dask.compute(*results, scheduler="processes", num_workers=num_workers)
        return np.sum( result )
    

    def overlap_cost_function(self, theta, theta_overlap):
        result_probs = self.node_overlap(theta = theta, theta_overlap=theta_overlap)
        aux1 = result_probs[:self.qubits]
        aux2 = result_probs[self.qubits:2*self.qubits]
        expval1 = []
        expval2 = []

        for i, term in enumerate(aux1):
            expval1.append( term[0] )
            expval2.append( aux2[i][0] )
        return 2*np.abs( 0.5 - np.sum( np.array(expval1)*np.array(expval2) )  )



'''
Clase con las funciones de coste para utilizar VQE y VQD
en un hamiltoniano de espines
'''
class vqe_spin():
    hamiltonian_object = None
    groups_caractericts = None
    coeff_object = None
    parity_terms = None

    def __init__(self, params):
        self.qubits = params['sites']
        self.spin = params['spin']
        self.correction = math.ceil( (int( 2*self.spin+1 ))/2  )

        terms = []
        coeff = []
        self.hamiltonian_object = None

        if params["pattern"] in ("open", "close"):
            for i in range(self.qubits-1):
                Xterm = ["I"]*self.qubits
                Yterm = ["I"]*self.qubits
                Zterm = ["I"]*self.qubits

                Xterm[i] = "X"; Xterm[i+1]= "X"
                Yterm[i] = "Y"; Yterm[i+1]= "Y"
                Zterm[i] = "Z"; Zterm[i+1]= "Z"

                terms.extend( [qml.pauli.string_to_pauli_word(list_to_string(Xterm)),
                               qml.pauli.string_to_pauli_word(list_to_string(Yterm)),
                               qml.pauli.string_to_pauli_word(list_to_string(Zterm))] )

                coeff.extend( [-params["exchange"][0], -params["exchange"][1], -params["exchange"][2]] )
                
            if params["pattern"] == "open":
                pass

            else:
                Xterm = ["I"]*self.qubits
                Yterm = ["I"]*self.qubits
                Zterm = ["I"]*self.qubits

                Xterm[0] = "X"; Xterm[self.qubits-1]= "X"
                Yterm[0] = "Y"; Yterm[self.qubits-1]= "Y"
                Zterm[0] = "Z"; Zterm[self.qubits-1]= "Z"

                terms.extend( [qml.pauli.string_to_pauli_word(list_to_string(Xterm)),
                               qml.pauli.string_to_pauli_word(list_to_string(Yterm)),
                               qml.pauli.string_to_pauli_word(list_to_string(Zterm))] )

                coeff.extend( [-params["exchange"][0], -params["exchange"][1], -params["exchange"][2]] )
        else:
            raise("Pattern no available, consider open and close pattern")
        

        terms, coeff = qml.pauli.group_observables(observables=terms,coefficients=coeff, grouping_type='qwc', method='rlf')
        Pauli_terms = []
        for group in terms:
            aux = []
            for term in group:
                string = Pauli_function(term, self.qubits)
                aux.append(string)
            Pauli_terms.append(aux)

        self.hamiltonian_object = Pauli_terms
        self.coeff_object = coeff
        self.parity_terms = np.array([ parity(i, self.spin, self.qubits) for i in range(2**(self.qubits*self.correction)) ]) 
        return


    def set_group_characteristics(self):
        aux_char = []
        for group in self.hamiltonian_object:
            aux_char.append( group_string(group) )
        self.groups_caractericts = aux_char
        return
    
    
    def process_group(self, theta, i):
        expval = []
        term = self.hamiltonian_object[i]
        charac = self.groups_caractericts[i]

        result_probs = self.node(theta=theta, obs=term, characteristic=charac)
        for k, probs in enumerate(result_probs):
            if is_identity(term[k]):
                expval.append( np.sum( probs ) )
            else:
                expval.append( np.sum( probs * self.parity_terms[:probs.shape[0]] ) )

        result = self.coeff_object[i]*np.array(expval)
        return np.sum(result)
    

    def cost_function(self, theta):
        results = []
        for i in range(len(self.hamiltonian_object)):
            results.append( self.process_group(theta, i) )
        return np.sum( results )


    def cost_function_parallel(self, theta):
        results = []
        for i in range(len(self.hamiltonian_object)):
            results.append( dask.delayed(self.process_group)(theta, i) )
        num_workers = 4
        result = dask.compute(*results, scheduler="processes", num_workers=num_workers)
        return np.sum( result )
    

    def overlap_cost_function(self, theta, theta_overlap):
        result_probs = self.node_overlap(theta = theta, theta_overlap=theta_overlap)
        return result_probs[0]
    

'''
Clase con las funciones de coste para utilizar VQE y VQD
en un hamiltoniano de espines
'''
class vqe_fermihubbard():
    hamiltonian_object= None
    hopping = 0.0
    potential = 0.0
    spin = 0.5
    qubits = 0

    def __init__(self, params):
        self.qubits = params["sites"]*2
        self.hopping = -params["hopping"]
        self.potential = params["potential"]
        fermi_sentence = 0.0

        if params["sites"] == 1:
            fermi_sentence +=  self.potential*FermiC(0)*FermiA(0)*FermiC(1)*FermiA(1)
        else:
            for i in range(params["sites"]-1):
                fermi_sentence +=  self.hopping*FermiC(2*i)*FermiA(2*i +2) + self.hopping*FermiC(2*i +2)*FermiA(2*i)
                fermi_sentence +=  self.hopping*FermiC(2*i+1)*FermiA(2*i +3) + self.hopping*FermiC(2*i +3)*FermiA(2*i +1)  
                fermi_sentence +=  self.potential*FermiC(2*i)*FermiA(2*i)*FermiC(2*i +1)*FermiA(2*i +1)

            if params["pattern"] == "close" and params["sites"] != 2:
                qsite = 2*(params["sites"]-1)
                fermi_sentence +=  self.hopping*FermiC(0)*FermiA(qsite) + self.hopping*FermiC(qsite)*FermiA(0)
                fermi_sentence +=  self.hopping*FermiC(1)*FermiA(qsite+1) + self.hopping*FermiC(qsite+1)*FermiA(1) 

        coeff, terms = qml.jordan_wigner( fermi_sentence, ps=True).hamiltonian().terms()
        terms, coeff = qml.pauli.group_observables(observables=terms,coefficients=np.real(coeff), grouping_type='qwc', method='rlf')
        Pauli_terms = []
        for group in terms:
            aux = []
            for term in group:
                string = Pauli_function(term, self.qubits)
                aux.append(string)
            Pauli_terms.append(aux)

        self.hamiltonian_object = Pauli_terms
        self.coeff_object = coeff
        self.parity_terms = np.array([ parity(i, self.spin, self.qubits) for i in range(2**self.qubits) ])
        return
    
    
    def set_group_characteristics(self):
        aux_char = []
        for group in self.hamiltonian_object:
            aux_char.append( group_string(group) )
        self.groups_caractericts = aux_char
        return
    

    def process_group(self, theta, i):
        expval = []
        term = self.hamiltonian_object[i]
        charac = self.groups_caractericts[i]

        result_probs = self.node(theta=theta, obs=term, characteristic=charac)
        for k, probs in enumerate(result_probs):
            if is_identity(term[k]):
                expval.append( np.sum( probs ) )
            else:
                expval.append( np.sum( probs * self.parity_terms[:probs.shape[0]] ) )

        result = self.coeff_object[i]*np.array(expval)
        return np.sum(result)
    

    def cost_function(self, theta):
        results = []
        for i in range(len(self.hamiltonian_object)):
            results.append( self.process_group(theta, i) )
        return np.sum( results )


    def cost_function_parallel(self, theta):
        results = []
        for i in range(len(self.hamiltonian_object)):
            results.append( dask.delayed(self.process_group)(theta, i) )
        num_workers = 4
        result = dask.compute(*results, scheduler="processes", num_workers=num_workers)
        return np.sum( result )
    
    
    def overlap_cost_function(self, theta, theta_overlap):
        result_probs = self.node_overlap(theta = theta, theta_overlap=theta_overlap)
        aux1 = result_probs[:self.qubits]
        aux2 = result_probs[self.qubits:2*self.qubits]
        expval1 = []
        expval2 = []

        for i, term in enumerate(aux1):
            expval1.append( term[0] )
            expval2.append( aux2[i][0] )
        return 2*np.abs( 0.5 - np.sum( np.array(expval1)*np.array(expval2) )  )