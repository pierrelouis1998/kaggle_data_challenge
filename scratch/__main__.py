import pickle
from pathlib import Path
from typing import Tuple, List

import networkx as nx
import numpy as np
import pandas as pd
from grakel import NeighborhoodHash, WeisfeilerLehman, HadamardCode, graph_from_networkx
from networkx import Graph
from sklearn.svm import SVC

from kaggle_data_challenge.scratch.rmkl import RMKL
from kaggle_data_challenge.scratch.svc import SupportVectorClassification
from kaggle_data_challenge.utils.load import load_my_data

KERNEL_SAVE_PATH = "./test_kernel_2"

kernels_loading_list = [
    {"name": "neighborhood_hash", "compute": NeighborhoodHash(normalize=True)},

    # {"name": "neighborhood_hash", "compute": NeighborhoodHash(R=4, nh_type="count_sensitive")},
    {"name": "weisfeiler_lehman", "compute": WeisfeilerLehman(normalize=True)},
    # {"name": "hadamard_code", "compute": HadamardCode(normalize=True)},
]


def normalize_kernels(train_kernel_matrices, test_kernel_matrices):
    normalized_train_kernel_matrices = []
    normalized_test_kernel_matrices = []
    for train_kernel_matrix, test_kernel_matrix in zip(train_kernel_matrices, test_kernel_matrices):
        normalized_train_kernel_matrix = train_kernel_matrix.astype(float)
        normalized_test_kernel_matrix = test_kernel_matrix.astype(float)

        norm = np.linalg.norm(train_kernel_matrix)
        normalized_train_kernel_matrix /= norm
        normalized_test_kernel_matrix /= norm

        normalized_train_kernel_matrices.append(normalized_train_kernel_matrix)
        normalized_test_kernel_matrices.append(normalized_test_kernel_matrix)

    return normalized_train_kernel_matrices, normalized_test_kernel_matrices


def load_kernel(name: str) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(f"{KERNEL_SAVE_PATH}/{name}.npy"), np.load(f"{KERNEL_SAVE_PATH}/{name}_test.npy")


def compute_all_kernels(train_data, test_data, save=True):
    # Need to transform data to Grakel for now
    train_data = list(graph_from_networkx(train_data, node_labels_tag="labels", edge_labels_tag="labels"))
    test_data = list(graph_from_networkx(test_data, node_labels_tag="labels", edge_labels_tag="labels"))

    train_kernel_matrices = []
    test_kernel_matrices = []
    for kernel_info in kernels_loading_list:
        kernel_compute = kernel_info['compute']
        train_kernel_matrix = kernel_compute.fit_transform(train_data)
        test_kernel_matrix = kernel_compute.transform(test_data)

        if save:
            name = kernel_info["name"]
            np.save(f"{KERNEL_SAVE_PATH}/{name}.npy", train_kernel_matrix)
            np.save(f"{KERNEL_SAVE_PATH}/{name}_test.npy", test_kernel_matrix)

        train_kernel_matrices.append(train_kernel_matrix)
        test_kernel_matrices.append(test_kernel_matrix)
    return train_kernel_matrices, test_kernel_matrices


def load_all_kernels():
    train_kernel_matrices = []
    test_kernel_matrices = []
    for kernel in kernels_loading_list:
        train_kernel_matrix, test_kernel_matrix = load_kernel(kernel['name'])

        train_kernel_matrices.append(train_kernel_matrix)
        test_kernel_matrices.append(test_kernel_matrix)

    return train_kernel_matrices, test_kernel_matrices


def load_dataset(data_dir=Path("data")) -> Tuple[List[Graph], List[str], List[Graph]]:
    with open(data_dir / "training_data.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(data_dir / "training_labels.pkl", 'rb') as f:
        train_labels = pickle.load(f)
    with open(data_dir / "test_data.pkl", 'rb') as f:
        test_data = pickle.load(f)

    return train_data, train_labels, test_data


def main(compute_kernels=True):
    # Load data
    train_data, labels, test_data = load_my_data()

    # Fix node and edge labels
    for graph in train_data + test_data:
        attributes = nx.get_node_attributes(graph, "labels")
        new_attributes = {key: attributes[key][0] for key in attributes.keys()}
        nx.set_node_attributes(graph, new_attributes, "labels")

        attributes = nx.get_edge_attributes(graph, "labels")
        new_attributes = {key: attributes[key][0] for key in attributes.keys()}
        nx.set_edge_attributes(graph, new_attributes, "labels")

    # Load or compute kernel on graphs
    if compute_kernels:
        train_kernel_matrices, test_kernel_matrices = compute_all_kernels(train_data, test_data)
    else:
        train_kernel_matrices, test_kernel_matrices = load_all_kernels()

    # train_kernel_matrices, test_kernel_matrices = normalize_kernels(train_kernel_matrices, test_kernel_matrices)

    # Load and fit Model (Kernel Learning)
    model = RMKL(max_iter=30, lr=.05, C=0.025)
    model.fit(train_kernel_matrices, labels)

    train_learnt_kernel = model.get_kernel_matrix()
    test_learnt_kernel = model.sum_kernels(test_kernel_matrices, model.weights)
    print(model.weights)

    # TODO Load and fit Model (Prediction)
    # svm = SupportVectorClassification()

    # TODO Remove this part
    svc = SVC(kernel="precomputed", probability=True, C=0.025)
    svc.fit(train_learnt_kernel, labels)
    y_pred = svc.predict_proba(test_learnt_kernel)
    y_pred_logit = np.log(y_pred[:, 1] / y_pred[:, 0])


    # Predict
    # y_pred = model.predict(test_kernel_matrices)

    # Fake logits for submission
    # y_pred_logit = np.zeros_like(y_pred)
    # y_pred_logit[y_pred < 0.5] = -1.
    # y_pred_logit[y_pred >= 0.5] = 1.

    # Write logits to csv
    Yte = {'Predicted': y_pred_logit}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv('pred_last_last.csv', index_label='Id')



if __name__ == '__main__':
    main(compute_kernels=False)
