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