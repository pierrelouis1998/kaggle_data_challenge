from typing import List

from grakel import graph_from_networkx

from kaggle_data_challenge.utils.load import load_my_data
import numpy as np
from tqdm import tqdm
from grakel import GraphKernel


def main():
    kernels = [
        {"name": "random_walk", "with_labels": False},
        {"name": "random_walk", "with_labels": True},
        {"name": "shortest_path", "with_labels": False},
        {"name": "shortest_path", "with_labels": True},
        {"name": "graphlet_sampling"},
        {"name": "subgraph_matching"},
        {"name": "lovasz_theta"},
        {"name": "svm_theta"},
        {"name": "neighborhood_hash"},
        {"name": "neighborhood_subgraph_pairwise_distance"},
        {"name": "odd_sth"},
        {"name": "pyramid_match"},
        {"name": "vertex_histogram"},
        {"name": "subtree_wl"},
        {"name": "edge_histogram"},
        {"name": "graph_hopper"},
        {"name": "weisfeiler_lehman"},
        {"name": "hadamard_code"},
    ]

    train_data, train_labels, test_data = load_my_data()

    train_data_grakel = list(graph_from_networkx(train_data, node_labels_tag="labels", edge_labels_tag="labels"))

    for graph in train_data_grakel:
        attributes = graph[1]
        new_attributes = {key: attributes[key][0] for key in attributes.keys()}
        graph[1] = new_attributes

        edge_attributes = graph[2]
        new_edge_attributes = {key: edge_attributes[key][0] for key in edge_attributes.keys()}
        graph[2] = new_edge_attributes

    for kernel in tqdm(kernels):
        try:
            kernel_computer = GraphKernel(kernel=kernel, n_jobs=-1)
            kernel_matrix = kernel_computer.fit_transform(train_data_grakel)
            if kernel.get("with_labels"):
                np.save(f"kernel/{kernel['name']}_with_labels.npy", kernel_matrix)
            else:
                np.save(f"kernel/{kernel['name']}.npy", kernel_matrix)
        except Exception as e:
            print(f"Couldn't compute {kernel['name']}: {e}")


if __name__ == '__main__':
    main()
