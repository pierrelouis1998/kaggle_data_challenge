import networkx as nx
import numpy as np
from grakel import graph_from_networkx, Graph, NeighborhoodHash
from numpy.random import RandomState
from sklearn.svm import SVC

from kaggle_data_challenge.pierre.mkl import load_kernel
from kaggle_data_challenge.scratch.neighborhood_hash_kernel import NeighborhoodHashKernel
from kaggle_data_challenge.scratch.svc import SupportVectorClassification
# from kaggle_data_challenge.scratch.svc import SupportVectorClassification
# from kaggle_data_challenge.scratch.svc3 import SVC3
from kaggle_data_challenge.utils.load import load_my_data

def main_compare_nx_and_grakel():
    train_data, train_labels, test_data = load_my_data()

    train_data_grakel = list(graph_from_networkx(train_data, node_labels_tag="labels", edge_labels_tag="labels"))
    test_data_grakel = list(graph_from_networkx(test_data, node_labels_tag="labels", edge_labels_tag="labels"))

    for graph in train_data_grakel:
        attributes = graph[1]
        new_attributes = {key: attributes[key][0] for key in attributes.keys()}
        graph[1] = new_attributes

        edge_attributes = graph[2]
        new_edge_attributes = {key: edge_attributes[key][0] for key in edge_attributes.keys()}
        graph[2] = new_edge_attributes

    for graph in train_data:
        attributes = nx.get_node_attributes(graph, "labels")
        new_attributes = {key: attributes[key][0] for key in attributes.keys()}
        nx.set_node_attributes(graph, new_attributes, "labels")

    graph_grakel_raw = train_data_grakel[0]
    graph_grakel = Graph(graph_grakel_raw[0], graph_grakel_raw[1])
    graph_nx = train_data[0]

    print(list(graph_grakel.get_vertices(purpose="any")))
    print(graph_grakel.get_labels(purpose="any"))
    print({n: graph_grakel.neighbors(n, purpose="any") for n in graph_grakel.get_vertices(purpose="any")})
    print(graph_nx.nodes)
    print(nx.get_node_attributes(graph_nx, "labels"))
    print({n: list(graph_nx.neighbors(n)) for n in graph_nx.nodes})

def main():
    train_data, train_labels, test_data = load_my_data()
    train_data = train_data[:10]

    train_data_grakel = list(graph_from_networkx(train_data, node_labels_tag="labels", edge_labels_tag="labels"))
    test_data_grakel = list(graph_from_networkx(test_data, node_labels_tag="labels", edge_labels_tag="labels"))

    for graph in train_data_grakel:
        attributes = graph[1]
        new_attributes = {key: attributes[key][0] for key in attributes.keys()}
        graph[1] = new_attributes

        edge_attributes = graph[2]
        new_edge_attributes = {key: edge_attributes[key][0] for key in edge_attributes.keys()}
        graph[2] = new_edge_attributes

    for graph in train_data:
        attributes = nx.get_node_attributes(graph, "labels")
        new_attributes = {key: attributes[key][0] for key in attributes.keys()}
        nx.set_node_attributes(graph, new_attributes, "labels")

    random_state = RandomState(42)
    nhk = NeighborhoodHashKernel(R=3, random_state=random_state)
    random_state = RandomState(42)
    nh = NeighborhoodHash(R=3, random_state=random_state, bits=64)

    nh.fit(train_data_grakel)
    nhk.fit(train_data)

    print(nh.X)
    print(nhk.X)

def main_svc():

    train_data, train_labels, test_data = load_my_data()
    kernel = load_kernel("neighborhood_hash")


    # csvc = SupportVectorClassification(C=1.0, kernel_matrix=kernel, labels=train_labels)
    # csvc.fit()

    # print(csvc.dual_coef_)
    # print(csvc.support_indices)
    own_scv = SupportVectorClassification()
    own_scv.fit(kernel, train_labels)
    print(own_scv.alphas)
    print(own_scv.support_vectors_)

    svc = SVC(kernel="precomputed")
    svc.fit(kernel, train_labels)
    print(svc.dual_coef_)
    print(svc.support_)

    # svc3 = SVC3(kernel, train_labels)
    # svc3.fit()
    # svm = SVM(kernel=kernel, C=1.0)
    # svm.fit(None, train_labels)

    print()

if __name__ == '__main__':
    main_svc()