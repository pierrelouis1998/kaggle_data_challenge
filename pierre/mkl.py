import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from MKLpy.algorithms import GRAM, EasyMKL, PWMK, MEMO
from MKLpy.scheduler import ReduceOnWorsening
from MKLpy.callbacks import EarlyStopping

from kaggle_data_challenge.utils.load import load_my_data

kernels = [
    # {"name": "random_walk", "with_labels": False},
    # {"name": "random_walk", "with_labels": True},
    # {"name": "shortest_path", "with_labels": False},
    # {"name": "shortest_path", "with_labels": True},
    # {"name": "graphlet_sampling"},
    # {"name": "subgraph_matching"},
    # {"name": "lovasz_theta"},
    {"name": "svm_theta"},
    {"name": "neighborhood_hash"},
    # {"name": "neighborhood_subgraph_pairwise_distance"},
    # {"name": "odd_sth"},
    # {"name": "pyramid_match"},
    # {"name": "vertex_histogram"},
    # {"name": "subtree_wl"},
    # {"name": "edge_histogram"},
    # {"name": "graph_hopper"},
    {"name": "weisfeiler_lehman"},
    {"name": "hadamard_code"},
]


def load_kernel(name: str, with_label: bool = False) -> np.ndarray:
    if with_label:
        return np.load(f"kernel/{name}_with_labels.npy")
    else:
        return np.load(f"kernel/{name}.npy")


def load_all_kernels():
    kernel_matrices = []
    for kernel in kernels:
        try:
            kernel_matrices.append(load_kernel(kernel['name'], kernel.get('with_labels', False)))
        except:
            pass
    # normalized_kernels = normalize_kernels(kernel_matrices)

    return kernel_matrices


def normalize_kernels(kernel_matrices):
    normalized_kernels = []

    for kernel_matrix in kernel_matrices:
        normalized_kernel = np.zeros((kernel_matrix.shape[0], kernel_matrix.shape[1]))
        for row in range(kernel_matrix.shape[0]):
            for col in range(kernel_matrix.shape[0]):
                if kernel_matrix[row, row] < 1e-5 or kernel_matrix[col, col] < 1e-5:
                    normalized_kernel[row, col] = 1e5
                else:
                    normalized_kernel[row, col] = kernel_matrix[row, col] = (
                                (kernel_matrix[row, row] ** 0.5) * (kernel_matrix[col, col] ** 0.5))

        normalized_kernels.append(normalized_kernel)

    return normalized_kernels


def main():
    train_data, labels, _ = load_my_data()

    n_samples = len(labels)
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.15)

    kernel_matrices = load_all_kernels()
    kernel_train_matrices = []
    kernel_test_matrices = []

    for kernel_matrix in kernel_matrices:
        kernel_train_matrices.append(kernel_matrix[train_idx, :][:, train_idx])
        kernel_test_matrices.append(kernel_matrix[test_idx, :][:, train_idx])
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    # Gram
    # kernel_train_matrices_torch = torch.from_numpy(np.array(kernel_train_matrices))
    # earlystop = EarlyStopping(kernel_test_matrices, test_labels, patience=5, cooldown=1, metric='roc_auc')
    # scheduler = ReduceOnWorsening()
    # mkl = GRAM(
    #     max_iter=1000,
    #     learning_rate=.01,
    #     callbacks=[earlystop],
    #     scheduler=scheduler).fit(kernel_train_matrices_torch, train_labels)

    # Easy
    # mkl = EasyMKL(lam=1.)
    # mkl = mkl.fit(kernel_train_matrices, train_labels)

    # PWNK
    # kernel_train_matrices_torch = torch.from_numpy(np.array(kernel_train_matrices))
    # cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # mkl = PWMK(delta=0, cv=cv).fit(kernel_train_matrices_torch, train_labels)

    # MEMO
    earlystop = EarlyStopping(
        kernel_test_matrices, test_labels,  # validation data, KL is a validation kernels list
        patience=5,  # max number of acceptable negative steps
        cooldown=1,  # how ofter we run a measurement, 1 means every optimization step
        metric='roc_auc',  # the metric we monitor
    )

    # ReduceOnWorsening automatically redure the learning rate when a worsening solution occurs
    scheduler = ReduceOnWorsening()

    mkl = MEMO(
        max_iter=1000,
        learning_rate=.1,
        callbacks=[earlystop],
        scheduler=scheduler).fit(kernel_train_matrices, train_labels)

    y_pred = mkl.predict(kernel_test_matrices)
    accuracy = accuracy_score(test_labels, y_pred)

    print(f"Accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    main()
