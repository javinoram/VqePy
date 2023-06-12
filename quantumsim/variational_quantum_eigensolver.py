from quantumsim.functions.ansatz import *
from quantumsim.functions.constans import *
from pennylane import qchem
import math

'''
Clases que representan implementaciones del metodo variational quantum eigensolver (VQE).

Cada clase representa un tipo concreto de hamiltoniano en el que puede aplicar VQE,
dentro de cada uno estan los metodos necesarios para poder contruir el paso de optimizacion.
(hamiltoniano, ansatz y funcion de coste)
'''

class variational_quantum_eigensolver_electronic(given_ansatz):
    hamiltonian_object= None
    symbols = None
    coordinates = None
    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'

    '''
    Iniciador de la clase que construye el hamiltoniano molecular, todas las variables
    son guardadas en la clase
    input:
        symbols: list [string]
        coordinates: list [float]
        params: dict
    return:
        result: none
    '''
    def __init__(self, symbols, coordinates, params= None):
        self.symbols = symbols
        self.coordinates = coordinates
        if params['mapping']:
            if params['mapping'] in ("jordan_wigner", "bravyi_kitaev"):
                self.mapping = params['mapping']
            else:
                raise Exception("Mapping no valido, considere jordan_wigner o bravyi_kitaev")
            
        if params['charge']:
            self.charge = params['charge']

        if params['mult']:
            self.mult = params['mult']

        if params['basis']:
            if params['basis'] in ("sto-3g", "6-31g", "6-311g", "cc-pvdz"):
                self.basis = params['basis']
            else:
                raise Exception("Base no valida, considere sto-3g, 6-31g, 6-311g, cc-pvdz")
        
        if params['method']:
            if params['method'] in ("pyscf", "dhf"):
                self.method = params['method']
            else:
                raise Exception("Metodo no valido, considere dhf o pyscf")

        
        aux_h, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates/2,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method)
        
        coeff, expression = aux_h.terms()
        Pauli_terms = []

        for k, term in enumerate(expression):
            auxiliar_string = ["I" for _ in range(self.qubits)]

            if type(term)==qml.ops.qubit.non_parametric_ops.PauliZ:
                index = term.wires[0]
                auxiliar_string[index] = "Z"
            
            elif type(term)==qml.ops.qubit.non_parametric_ops.PauliX:
                index = term.wires[0]
                auxiliar_string[index] = "X"
            
            elif type(term)==qml.ops.qubit.non_parametric_ops.PauliY:
                index = term.wires[0]
                auxiliar_string[index] = "Y"

            elif type(term)==qml.ops.identity.Identity:
                pass
            
            else:
                Nonidentical = term.non_identity_obs
                for pauli in Nonidentical:
                    index = pauli.wires[0]
                    if type(pauli)==qml.ops.qubit.non_parametric_ops.PauliZ:
                        auxiliar_string[index] = "Z"
                    elif type(pauli)==qml.ops.qubit.non_parametric_ops.PauliX:
                        auxiliar_string[index] = "X"
                    elif type(pauli)==qml.ops.qubit.non_parametric_ops.PauliY:
                        auxiliar_string[index] = "Y"
                    else:
                        pass
            string = ""
            for s in auxiliar_string:
                string+= s
            Pauli_terms.append([coeff[k],string])
        self.hamiltonian_object = Pauli_terms
        del aux_h, coeff, expression
        return
    
    '''
    Funcion de coste: Funcion de coste que utiliza la funcion de valor esperado de qml
    input:
        theta: list [float]
    return:
        result: float
    '''
    def cost_function(self, theta):
        params = [theta[:self.repetition], theta[self.repetition:]]
        result = 0.0
        for coeff, term in self.hamiltonian_object:
            aux = coeff
            aux.requires_grad = True
            
            if is_identity(term):
                result += aux
            else:
                result_probs = self.node(theta = params, obs = term)
                for i in range(len(result_probs)):
                    result += aux*result_probs[i]*parity(i)
        return result


class variational_quantum_eigensolver_spin(spin_ansatz):
    hamiltonian_object = None
    hamiltonian_index = []

    '''
    Iniciador de la clase que construye la matriz de indices, todas las variables
    son guardadas en la clase
    input:
        params: dict
    return:
        result: none
    '''
    def __init__(self, params):
        self.qubits = len(params['pauli_string'][0][0])
        self.spin = params['spin']
        self.correction = math.ceil( (int( 2*self.spin+1 ))/2  )
        for term in params['pauli_string']:
            aux = []
            for i, string in enumerate(term[0]):
                if string != 'I': aux.append(i)
            
            if len(aux) != 2:
                raise Exception("Terminos del hamiltoniano tienen mas de 2 interacciones")
            
            self.hamiltonian_index.append(aux)
        self.hamiltonian_object = params['pauli_string']
        return
    

    '''
    Funcion de coste: Funcion de coste usando descomposicion de suma de probabilidades
    input:
        theta: list [float]
    return:
        result: float
    '''
    def cost_function(self, theta):
        ansatz = theta
        result= 0.0
        for i, term in enumerate(self.hamiltonian_index):
            result_term = self.node( theta=ansatz, obs= term, pauli= self.hamiltonian_object[i][0])
            exchange = self.hamiltonian_object[i][1]
            for s in conts_spin[str(self.spin)]["2"]:
                index = int(s, 2)
                result += exchange*result_term[index]
        return result
    

'''
    1D Hubbard model, lineal
    Cada sitio es modelado usando dos qubits.
    Estos son ordenados agrupandolos segun el espin del sitio
    [sitios de espin up]_n [sitios de espin down]_n
'''
class variational_quantum_eigensolver_fermihubbard(given_fermihubbard_ansazt):
    hamiltonian_object= None

    hopping_index = []
    hopping = 0.0

    potential_index = []
    potential = 0.0
    
    def __init__(self, params):
        self.qubits = params["sites"]*2
        self.hamiltonian_object = None
        self.hopping = params["hopping"]
        self.potential = params["potential"]
        for i in range( params["sites"] ):
            index = [ i,  params["sites"] + i]
            self.potential_index.append(index)

        for i in range( params["sites"]-1 ):
            index1 = [ i,  i+1 ]
            index2 = [ params["sites"]+i,  params["sites"]+ i+1 ]
            self.hopping_index.append(index1)
            self.hopping_index.append(index2)

        print(self.hopping_index)
        print(self.potential_index)
        return
    
    '''
    Funcion de coste: Funcion de coste que utiliza la funcion de valor esperado de qml
    input:
        theta: list [float]
    return:
        result: float
    '''
    def cost_function(self, theta):
        params = [theta[:self.repetition], theta[self.repetition:]]
        result = 0.0
        #Onsite interaccion
        for term in self.potential_index:
            result_aux = self.node(theta = params, obs = term, type='U', pauli="I")
            result+= -self.potential*result_aux[3]
        #result = self.node(theta = params, obs = self.hamiltonian_object)
        
        #Hopping interaction
        for term in self.hopping_index:
            result_X = self.node(theta = params, obs = term, type='t', pauli="X")
            result_Y = self.node(theta = params, obs = term, type='t', pauli="Y")

            result+= -self.hopping*(result_X[0]+result_X[3]-result_X[1]-result_X[2])
            result+= -self.hopping*(result_Y[0]+result_Y[3]-result_Y[1]-result_Y[2])
        return result