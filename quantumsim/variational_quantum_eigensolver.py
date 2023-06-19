from quantumsim.functions.ansatz import *
from quantumsim.functions.constans import *
from pennylane import qchem
import math

'''
Clases que representan implementaciones del metodo variational quantum eigensolver (VQE).
'''

class variational_quantum_eigensolver_electronic(given_ansatz):
    hamiltonian_object= None
    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'

    '''
    Iniciador de la clase que construye el hamiltoniano molecular y lo almacena en
    la representación Pauli String.
    input:
        symbols: list [string]
        coordinates: list [float]
        params: dict
    return:
        result: none
    '''
    def __init__(self, symbols, coordinates, params= None):
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

        self.hamiltonian_object = conmute_group(Pauli_terms)
        del aux_h, coeff, expression, Pauli_terms
        return
    
    '''
    Funcion de coste:
    input:
        theta: Lista de parametros ([float])
    return:
        result: Valor esperado (float)
    '''
    def cost_function(self, theta):
        params = [theta[:self.repetition], theta[self.repetition:]]
        result = 0.0

        #Iteracion sobre grupos conmutantes
        for group in self.hamiltonian_object:
            if is_identity(group[0][1]):
                exchange = group[0][0]
                exchange.requires_grad = True
                result += exchange
            else:
                result_probs = self.node(theta = params, obs = group)
                for k, probs in enumerate(result_probs):
                    exchange = group[k][0]
                    exchange.requires_grad = True
                    for j in range(len(probs)):
                        result += exchange*probs[j]*parity(j)
        return result


class variational_quantum_eigensolver_spin(spin_ansatz):
    hamiltonian_object = None
    hamiltonian_index = []

    '''
    Iniciador de la clase que construye el hamiltoniano de espines y lo almacena en
    la representación Pauli String.
    input:
        params: dict
    return:
        result: none
    '''
    def __init__(self, params):
        self.qubits = len(params['pauli_string'][0][1])
        self.spin = params['spin']
        self.correction = math.ceil( (int( 2*self.spin+1 ))/2  )
        self.hamiltonian_object = conmute_group(params['pauli_string'])
        return
    

    '''
    Funcion de coste: Funcion de coste usando descomposicion de suma de probabilidades
    input:
        theta: list [float]
    return:
        result: float
    '''
    def cost_function(self, theta):
        result= 0.0
        for group in self.hamiltonian_object:
            result_probs = self.node( theta=theta, obs=group)
            for k, probs in enumerate(result_probs):
                    aux = group[k][0]
                    for j in range(len(probs)):
                        result += aux*probs[j]*parity(j)
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
        
        #Hopping interaction
        for term in self.hopping_index:
            result_X = self.node(theta = params, obs = term, type='t', pauli="X")
            result_Y = self.node(theta = params, obs = term, type='t', pauli="Y")

            result+= self.hopping*(result_X[0]+result_X[3]-result_X[1]-result_X[2])
            result+= self.hopping*(result_Y[0]+result_Y[3]-result_Y[1]-result_Y[2])
        return result