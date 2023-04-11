from typing import List

from grakel import graph_from_networkx, NeighborhoodHash
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from kaggle_data_challenge.utils.load import load_my_data
import numpy as np
from tqdm import tqdm
from grakel import GraphKernel

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


def main_compute():
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
            kernel_computer = GraphKernel(kernel=kernel, n_jobs=-1, Nystroem=20)
            kernel_matrix = kernel_computer.fit_transform(train_data_grakel)
            if kernel.get("with_labels"):
                np.save(f"kernel/{kernel['name']}_with_labels.npy", kernel_matrix)
            else:
                np.save(f"kernel/{kernel['name']}.npy", kernel_matrix)
        except Exception as e:
            print(f"Couldn't compute {kernel['name']}: {e}")


def main_cross_validation():
    train_data, train_labels, _ = load_my_data()

    n_samples = len(train_labels)
    indices = np.arange(n_samples)
    indices_list = [train_test_split(indices, test_size=0.15, stratify=train_labels) for _ in range(10)]

    accuracies = {}
    for kernel in kernels:
        kernel_name = kernel['name']

        try:
            print(f"Loading {kernel_name}")
            if kernel.get("with_labels"):
                kernel_data = np.load(f"kernel/{kernel['name']}_with_labels.npy")
            else:
                kernel_data = np.load(f"kernel/{kernel['name']}.npy")
            # mask = np.isnan(kernel_data)
            # kernel_data[mask] = 0.
            # kernel_data[mask] = kernel_data.max()

            accuracy = 0.
            for (train_indices, test_indices) in indices_list:
                # print("Computing train and test kernels")
                kernel_train = kernel_data[train_indices, :][:, train_indices]
                kernel_test = kernel_data[test_indices, :][:, train_indices]

                # print("Computing SVC")
                clf = SVC(kernel='precomputed')
                clf.fit(kernel_train, train_labels[train_indices])
                y_pred = clf.predict(kernel_test)

                # print(kernel_name)
                # print("%2.2f %%" % (round(accuracy_score(train_labels[test_indices], y_pred) * 100)))
                # accuracy += accuracy_score(train_labels[test_indices], y_pred)
                fpr, tpr, _ = roc_curve(train_labels[test_indices], y_pred)
                accuracy += auc(fpr, tpr)
            accuracies[kernel['name']] = accuracy / len(indices_list)
            print(f"AOC: {accuracies[kernel['name']]:.2%}")
        except Exception as e:
            print(f"Couldn't load kernel {kernel_name}: {e}")

    for key in accuracies.keys():
        print(f"{key['name']}: {accuracies[key]:.2%}")


def main():
    train_data, train_labels, test_data = load_my_data()

    train_data_grakel = list(graph_from_networkx(train_data, node_labels_tag="labels", edge_labels_tag="labels"))

    for graph in train_data_grakel:
        attributes = graph[1]
        new_attributes = {key: attributes[key][0] for key in attributes.keys()}
        graph[1] = new_attributes

        edge_attributes = graph[2]
        new_edge_attributes = {key: edge_attributes[key][0] for key in edge_attributes.keys()}
        graph[2] = new_edge_attributes

    n_samples = len(train_labels)
    indices = np.arange(n_samples)
    indices_list = [train_test_split(indices, test_size=0.15, stratify=train_labels) for _ in range(1)]
    for r in tqdm([2, 4, 8, 16, 32]):
        for nh_type in ['count_sensitive', 'simple']:
            kernel_computer = NeighborhoodHash(n_jobs=16, R=r, nh_type=nh_type)
            kernel = kernel_computer.fit_transform(train_data_grakel)
            accuracy = 0.
            for (train_indices, test_indices) in indices_list:
                # print("Computing train and test kernels")
                kernel_train = kernel[train_indices, :][:, train_indices]
                kernel_test = kernel[test_indices, :][:, train_indices]

                # print("Computing SVC")
                clf = SVC(kernel='precomputed')
                clf.fit(kernel_train, train_labels[train_indices])
                y_pred = clf.predict(kernel_test)

                # print(kernel_name)
                # print("%2.2f %%" % (round(accuracy_score(train_labels[test_indices], y_pred) * 100)))
                fpr, tpr, _ = roc_curve(train_labels[test_indices], y_pred)
                accuracy += auc(fpr, tpr)
            print(f"Accuracy(R={r}, type={nh_type}): {accuracy / len(indices_list):.2%}")


if __name__ == '__main__':
    main_cross_validation()
