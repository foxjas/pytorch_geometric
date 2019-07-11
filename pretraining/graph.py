"""
"""

import sys
import networkx as nx

def readEdgelist(fPath, singletons=False, skip=0):
    """
    Read edge list, and returns networkx graph
    Currently only handles undirected graph

    singletons: whether to preserve singletons
    skip: number of header lines to skip
    """
    min_id = sys.maxsize
    max_id = -1 
    edges = []
    with open(fPath, 'r') as f:
        for _ in range(skip):
            f.next()

        for l in f:
            src, dst = map(int, l.strip().split(" "))    
            edges.append((src,dst))
            min_id = min(min_id, src, dst)
            max_id = max(max_id, src, dst)
    
    G = nx.Graph() # undirected
    G.add_edges_from(edges)
    if singletons:
        G.add_nodes_from(list(range(min_id, max_id+1)))

    print("Number of nodes in graph: {}".format(G.number_of_nodes()))
    print("Number of edges in graph: {}".format(G.number_of_edges()))
    return G


def relabelGraph(G):
    """
    Relabels node IDs to [0,...,|V|-1] range. 
    Returns list of pairs (old id,new id)
    """
    G_r = nx.convert_node_labels_to_integers(G, label_attribute='id_orig')
    old_new_ids = [(data['id_orig'], new_id) for new_id, data in G.nodes.data()]
    return old_new_ids


def graphToCOO(G):
    """
    Converts graph to edge list of dimension
        [|V|,2]
    Returns list of edges
    """
    edgelist = [(src, dst) for src, dst,_ in nx.to_edgelist(G)]
    return edgelist
