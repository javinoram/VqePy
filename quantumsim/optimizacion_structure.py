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

        _, self.qubits = qchem.molecular_hamiltonian(
            symbols= symbols,
            coordinates= coordinates/2,
            mapping= self.mapping,
            charge= self.charge,
            mult= self.mult,
            basis= self.basis,
            method= self.method)
        
        #coeff, expression = aux_h.terms()
        #Pauli_terms = []

        #for k, term in enumerate(expression):
        #    auxiliar_string =[]
        #    if type(term)==qml.ops.qubit.non_parametric_ops.PauliZ:
        #        index = term.wires[0]
        #        for j in range(self.qubits):
        #            if j== index:
        #                auxiliar_string.append("Z")
        #            else:
        #                auxiliar_string.append("I")
            
        #    elif type(term)==qml.ops.qubit.non_parametric_ops.PauliX:
        #        index = term.wires[0]
        #        for j in range(self.qubits):
        #            if j== index:
        #                auxiliar_string.append("X")
        #            else:
        #                auxiliar_string.append("I")
            
        #    elif type(term)==qml.ops.qubit.non_parametric_ops.PauliY:
        #        index = term.wires[0]
        #        for j in range(self.qubits):
        #            if j== index:
        #                auxiliar_string.append("Y")
        #            else:
        #                auxiliar_string.append("I")

        #    elif type(term)==qml.ops.identity.Identity:
        #        for j in range(self.qubits):
        #            auxiliar_string.append("I")
        #    else:
        #        Nonidentical = term.non_identity_obs
        #        auxiliar_string = ["I" for _ in range(self.qubits)]
        #        for pauli in Nonidentical:
        #            index = pauli.wires[0]
        #            if type(pauli)==qml.ops.qubit.non_parametric_ops.PauliZ:
        #                auxiliar_string[index] = "Z"
        #            elif type(pauli)==qml.ops.qubit.non_parametric_ops.PauliX:
        #                auxiliar_string[index] = "X"
        #            elif type(pauli)==qml.ops.qubit.non_parametric_ops.PauliY:
        #                auxiliar_string[index] = "Y"
        #            else:
        #                pass
        #    string = ""
        #    for s in auxiliar_string:
        #        string+= s
        #    Pauli_terms.append([coeff[k], string])
        #self.hamiltonian_object = conmute_group(Pauli_terms)
        #del aux_h, coeff, expression
        return

    def grad_x(self, theta, x):
        params = [theta[:len(self.singles)*self.repetition], theta[len(self.singles)*self.repetition:]]
        grad = []
        delta = 0.01

        for i in range(len(x)):
            shift = np.zeros_like(x)
            shift[i] += 0.5 * delta

            coeff, expression = ((self.H(x + shift) - self.H(x - shift)) * delta**-1).terms()
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
                Pauli_terms.append([coeff[k], string])
            groups = conmute_group(Pauli_terms)
            
            result = 0.0
            for j, group in enumerate(groups):
                if is_identity(group[0][1]):
                    aux = group[0][0]
                    aux.requires_grad = True
                    result += aux
                        
                else:
                    result_probs = self.node(theta = params, obs = [g[1] for g in group])
                    for k, probs in enumerate(result_probs):
                        aux = group[k][0]
                        aux.requires_grad = True

                        result_aux = 0.0
                        for j in range(len(probs)):
                            result_aux += probs[j]*parity(j)
                        result += result_aux*aux
                
            grad.append( result )
        return np.array(grad)
    
    #Retorna los coeficientes
    def H(self, x):
        return qml.qchem.molecular_hamiltonian(self.symbols, x, mult= self.mult, charge=self.charge)[0]
    
    def cost_function(self, theta, x):
        params = [theta[:len(self.singles)*self.repetition], theta[len(self.singles)*self.repetition:]]
        
        coeff, expression = self.H(x).terms()
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
            Pauli_terms.append([coeff[k], string])
        groups = conmute_group(Pauli_terms)

        result = 0.0
        for k, group in enumerate(groups):
            if is_identity(group[0][1]):
                aux = group[0][0]
                aux.requires_grad = True
                result += aux
            else:
                result_probs = self.node(theta = params, obs = [g[1] for g in group])
                for i, probs in enumerate(result_probs):
                    aux = group[i][0]
                    aux.requires_grad = True

                    aux_result = 0.0
                    for j in range(len(probs)):
                        aux_result += probs[j]*parity(j)
                    result += aux_result*aux
        return result