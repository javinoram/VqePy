import networkx as nx


'''
input:
    params: diccionario con los parametros de la lattice
        -tipo de lattice
        -tamaño: tupla
        -periodicidad
return:
    list of the graph's edges
'''
def lattice(params):

    periodicity = False
    if isinstance(params['bound'], list):
        periodicity = params['bound']
    elif params['bound'] == True:
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