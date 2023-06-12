from quantumsim.functions.ansatz import *
from quantumsim.functions.constans import *
from pennylane import qchem
from pennylane import numpy as np

class optimization_structure(given_ansatz):
    symbols = None
    coordinates = None
    hamiltonian_object = None
    exp_sub_hamil = []

    mapping= 'jordan_wigner'
    charge= 0
    mult= 1
    basis='sto-3g'
    method='dhf'

    def __init__(self, symbols, coordinates, params= None):
        self.symbols = symbols
        self.coordinates = coordinates

        if params['mapping']:
            if params['mapping'] in ("jordan_wigner", "bravyi_kitaev"):
                self.mapping = params['mapping']
            else:
                raise Exception("Mapping no valido, considere jordan_wigner o bravyi_kitaev")
            
        elif params['charge']:
            self.charge = params['charge']

        elif params['mult']:
            self.mult = params['mult']

        elif params['basis']:
            if params['basis'] in ("sto-3g", "6-31g", "6-311g", "cc-pvdz"):
                self.basis = params['basis']
            else:
                raise Exception("Base no valida, considere sto-3g, 6-31g, 6-311g, cc-pvdz")
        
        elif params['method']:
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
            auxiliar_string =[]
            if type(term)==qml.ops.qubit.non_parametric_ops.PauliZ:
                index = term.wires[0]
                for j in range(self.qubits):
                    if j== index:
                        auxiliar_string.append("Z")
                    else:
                        auxiliar_string.append("I")
            
            elif type(term)==qml.ops.qubit.non_parametric_ops.PauliX:
                index = term.wires[0]
                for j in range(self.qubits):
                    if j== index:
                        auxiliar_string.append("X")
                    else:
                        auxiliar_string.append("I")
            
            elif type(term)==qml.ops.qubit.non_parametric_ops.PauliY:
                index = term.wires[0]
                for j in range(self.qubits):
                    if j== index:
                        auxiliar_string.append("Y")
                    else:
                        auxiliar_string.append("I")

            elif type(term)==qml.ops.identity.Identity:
                for j in range(self.qubits):
                    auxiliar_string.append("I")
            
            else:
                Nonidentical = term.non_identity_obs
                auxiliar_string = ["I" for _ in range(self.qubits)]
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
            Pauli_terms.append(string)
        self.hamiltonian_object = Pauli_terms
        del aux_h, coeff, expression
        return

    def grad_x(self, theta, x):
        params = [theta[:len(self.singles)*self.repetition], theta[len(self.singles)*self.repetition:]]
        grad = []
        delta = 0.01

        exp_vals = []
        for term in self.hamiltonian_object:
            if is_identity(term):
                exp_vals.append(1)
            else:
                result_probs = self.node(theta = params, obs = term)
                result_aux = 0.0
                for i in range(len(result_probs)):
                    result_aux += result_probs[i]*parity(i)
                exp_vals.append(result_aux)

        for i in range(len(x)):
            shift = np.zeros_like(x)
            shift[i] += 0.5 * delta
            res = (self.H(x + shift) - self.H(x - shift)) * delta**-1
            grad.append( sum(np.array(exp_vals)*res) )
        return np.array(grad)
    
    #Retorna los coeficientes
    def H(self, x):
        return qml.qchem.molecular_hamiltonian(self.symbols, x, mult= self.mult, charge=self.charge)[0].terms()[0]
    
    def cost_function(self, theta, x):
        params = [theta[:len(self.singles)*self.repetition], theta[len(self.singles)*self.repetition:]]
        
        coeff = self.H(x)
        result = 0.0
        for k, term in enumerate(self.hamiltonian_object):
            aux = coeff[k]
            aux.requires_grad = True
            if is_identity(term):
                result += aux
            else:
                result_probs = self.node(theta = params, obs = term)
                for i in range(len(result_probs)):
                    result += aux*result_probs[i]*parity(i)
        return result