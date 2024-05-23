import networkx as nx


'''
Funcion que retorna las coordenadas de los nodos del grafo que contruyen la grilla del sistema
input:
    params: diccionario con los parametros de la lattice
return:
    edges: lista con las conexiones entre los nodos
    node: lista con los nodos del sistema
'''
def lattice(params):
    periodicity = params['bound']

    if params['lattice'] == 'chain':
        lattice = nx.grid_2d_graph(params['size'][0], params['size'][1], periodicity)

    elif params['lattice'] == 'triangle':
        lattice = nx.triangular_lattice_graph(params['size'][0], params['size'][1], periodicity)

    elif params['lattice'] == 'square':
        lattice = nx.grid_2d_graph(params['size'][0], params['size'][1], periodicity)
    
    elif params['lattice'] == 'hexagon':
        lattice = nx.hexagonal_lattice_graph(params['size'][0], params['size'][1], periodicity)

    return lattice.edges(), lattice.nodes()

"""
Funcion para construir una grilla custom
input:
    params: diccionario con los parametros de la lattice
        - node: numero de nodos en cada eje [X, Y]
        - edges: lista de conecciones entre los nodos (n1, n2)
return:
    edges: lista con las conexiones entre los nodos
    node: lista con los nodos del sistema
"""
def custom_lattice(params):
    nodes = []
    for i in range( params['node'][0] ):
        nodes = nodes + [ (i,j) for j in range( params['node'][1] ) ]
    
    edges = []
    for edge in params['edges']:
        x1 = edge[0]
        x2 = edge[1]
        
        t1 = ( x1%params['node'][0], x1//params['node'][0] )
        t2 = ( x2%params['node'][0], x2//params['node'][0] )
        edges.append( (t1, t2) )
    return  edges, nodes
