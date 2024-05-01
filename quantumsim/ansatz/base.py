import pennylane as qml
import jax

"""
Clase padre de los ansatz, donde se definen las caracteristicas y funciones
que todo ansatz debe tener.
"""
class base_ansatz():
    base = "default.qubit"
    backend = ""
    token = ""
    interface = "autograd"
    diff_method = "best"

    device= None
    node = None
    qubits = 0
    repetition = 0

    """
    Funcion para setear caracteristicas del entorno de ejecucion del circuito
    input:
        params: diccionario de parametros para inicializar el entorno de ejecucion 
        del circuito
    """
    def set_device(self, params) -> None:
        self.qubits = params["qubits"]
        self.base = params['base']

        ##Maquinas reales
        if self.base == 'qiskit.ibmq':
            try:
                self.backend = params['backend']
                self.token = params['token']
                self.device= qml.device(self.base, backend=self.backend, 
                    wires=self.qubits,  ibmqx_token= self.token)
            except KeyError:
                print( "Parametro no encontrado, recuerde agregar backend y token" )

        ##Simuladores de qiskit
        elif self.base == "qiskit.aer":
            try: 
                self.backend = params['backend']
                self.device= qml.device(self.base, backend=self.backend, wires=self.qubits)
                self.device= qml.device(self.base, backend=self.backend, wires=self.qubits)
            except KeyError:
                print( "Parametro no encontrado, recuerde agregar backend" )
    
        ##Simuladores de pennylane
        else:
            self.device= qml.device(self.base, wires=self.qubits)
    
    """
    Funcion para setear caracteristicas del ejecutor del circuito
    input:
        params: diccionario de parametros para inicializar el ejecutor del circuito
    """
    def set_node(self, params) -> None:
        if "pattern" in params:
            self.pattern = params["pattern"]
        
        if params['repetitions']:
            self.repetition = params['repetitions']
        
        if params['interface']:
            self.interface= params['interface']
        
        if params['diff_method']:
            self.diff_method = params['diff_method']
        
        self.node = qml.QNode(self.circuit_state, self.device, interface=self.interface, diff_method=self.diff_method)
        if self.interface == "jax" or self.interface == "jax-jit":
            self.node = jax.jit(self.node)
    

    """
    Funcion para obtener el estado al final del circuito, despues de aplicar el ansatz, esto 
    solo funciona con simuladores
    input:
        theta: vector de parametros del circuito
    output:
        estado: retorna el estado del circuito como un arreglo numpy
    """
    def get_state(self, theta):
        node = qml.QNode(self.circuit_state, self.device, interface=self.interface, diff_method=self.diff_method)
        if self.interface == "jax" or self.interface == "jax-jit":
            node = jax.jit(node)
        return node(theta)