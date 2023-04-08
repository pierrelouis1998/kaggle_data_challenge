import networkx as nx


def graph_product_node_labels(G1: nx.Graph, G2: nx.Graph):
    """
    Get the tensor product of G1 and G2 and get the subgraph
    such that (v1,v2) have the same labels
    :param G1: First graph
    :param G2: Second graph
    :return: G = G1 X G2
    """
    G = nx.tensor_product(G1, G2)
    new_nodes = []
    for node in G.nodes:
        if G.nodes[node]['labels'][0] == G.nodes[node]['labels'][1]:
            new_nodes.append(node)
    new_G = nx.subgraph(G, new_nodes)
    return new_G


def graph_product_nodes_and_edges(G1: nx.Graph, G2: nx.Graph):
    """
    Get the tensor product of G1 and G2 and get the subgraph
    such that (v1,v2) have the same labels and for every e = (v1,v2), (v1',v2'), (v1,v1')
    has same label as (v2,v2').
    :param G1: First Graph
    :param G2: Second graph
    :return: G = G1 X g2
    """
    raise NotImplementedError
