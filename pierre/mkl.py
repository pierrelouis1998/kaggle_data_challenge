import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split, KFold
from MKLpy.algorithms import GRAM, EasyMKL, PWMK, MEMO
from MKLpy.scheduler import ReduceOnWorsening
from MKLpy.callbacks import EarlyStopping
from sklearn.svm import SVC
from tqdm import tqdm

from kaggle_data_challenge.scratch.rmkl import RMKL
from kaggle_data_challenge.utils.load import load_my_data

kernels = [
    # {"name": "random_walk", "with_labels": False},
    # {"name": "random_walk", "with_labels": True},
    # {"name": "shortest_path", "with_labels": False},
    # {"name": "shortest_path", "with_labels": True},
    # {"name": "graphlet_sampling"},
    # {"name": "subgraph_matching"},
    # {"name": "lovasz_theta"},
    # {"name": "svm_theta"},
    {"name": "neighborhood_hash"},
    # {"name": "neighborhood_subgraph_pairwise_distance"},
    # {"name": "odd_sth"},
    # {"name": "pyramid_match"},
    # {"name": "vertex_histogram"},
    # {"name": "subtree_wl"},
    # {"name": "edge_histogram"},
    # {"name": "graph_hopper"},
    {"name": "weisfeiler_lehman"},
    # {"name": "hadamard_code"},
]


def load_kernel(name: str, with_label: bool = False) -> np.ndarray:
    if with_label:
        return np.load(f"test_kernel_2/{name}_with_labels.npy")
    else:
        return np.load(f"test_kernel_2/{name}.npy")


def load_all_kernels():
    kernel_matrices = []
    for kernel in kernels:
        try:
            kernel_matrices.append(load_kernel(kernel['name'], kernel.get('with_labels', False)))
        except:
            pass
    # normalized_kernels = normalize_kernels(kernel_matrices)
    normalize_kernels_2(kernel_matrices)

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

def normalize_kernels_2(kernel_matrices):
    for kernel_matrix in kernel_matrices:
        kernel_matrix = kernel_matrix.astype(float)
        kernel_matrix /= np.linalg.norm(kernel_matrix)
        # kernel_matrix += (1. * np.eye(kernel_matrix.shape[0]))


def main(Cs):
    train_data, labels, _ = load_my_data()

    n_samples = len(labels)
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=labels)

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
    # y_pred = mkl.predict(kernel_test_matrices)
    # fpr, tpr, _ = roc_curve(test_labels, y_pred)
    # auc_score = auc(fpr, tpr)

    # accuracy = accuracy_score(test_labels, y_pred)

    # print(f"AUC: {auc_score:.2%}")
    # PWNK
    # kernel_train_matrices_torch = torch.from_numpy(np.array(kernel_train_matrices))
    # cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # mkl = PWMK(delta=0, cv=cv).fit(kernel_train_matrices_torch, train_labels)

    # MEMO
    # kernel_train_matrices_torch = torch.from_numpy(np.array(kernel_train_matrices))
    # earlystop = EarlyStopping(
    #     kernel_test_matrices, test_labels,  # validation data, KL is a validation kernels list
    #     patience=5,  # max number of acceptable negative steps
    #     cooldown=1,  # how ofter we run a measurement, 1 means every optimization step
    #     metric='roc_auc',  # the metric we monitor
    # )

    # ReduceOnWorsening automatically redure the learning rate when a worsening solution occurs
    # scheduler = ReduceOnWorsening()
    #
    # mkl = MEMO(
    #     max_iter=100,
    #     learning_rate=0.1,
    #     solver="cvxopt",
    #     callbacks=[earlystop],
    #     scheduler=scheduler).fit(kernel_train_matrices_torch, train_labels)
    # y_pred = mkl.predict(kernel_test_matrices)
    # fpr, tpr, _ = roc_curve(test_labels, y_pred)
    # auc_score = auc(fpr, tpr)

    # accuracy = accuracy_score(test_labels, y_pred)

    # print(f"AUC: {auc_score:.2%}")

    # R-MKL
    # kernel_train_matrices_torch = torch.from_numpy(np.array(kernel_train_matrices))
    # earlystop = EarlyStopping(
    #     kernel_test_matrices, test_labels,  # validation data, KL is a validation kernels list
    #     patience=5,  # max number of acceptable negative steps
    #     cooldown=1,  # how ofter we run a measurement, 1 means every optimization step
    #     metric='roc_auc',  # the metric we monitor
    # )

    # ReduceOnWorsening automatically redure the learning rate when a worsening solution occurs
    scheduler = ReduceOnWorsening()

    for C in Cs:
        mkl = RMKL(max_iter=30, lr=.05, C=C)

        mkl.fit(kernel_train_matrices, train_labels)

        y_pred = mkl.predict(kernel_test_matrices)
        fpr, tpr, _ = roc_curve(test_labels, y_pred)
        auc_score = auc(fpr, tpr)

        svc = SVC(kernel="precomputed", probability=True, C=C)
        svc.fit(mkl.get_kernel_matrix(), train_labels)
        local_kernel_test_matrix = mkl.sum_kernels(kernel_test_matrices, mkl.weights)
        y_pred_2 = svc.predict_log_proba(local_kernel_test_matrix)
        y_pred_logit = y_pred_2[:, 1] - y_pred_2[:, 0]
        fpr, tpr, _ = roc_curve(test_labels, y_pred_logit)
        auc_score_2 = auc(fpr, tpr)

        # accuracy = accuracy_score(test_labels, y_pred)
        print(f"C = {C}")
        print(f"AUC: {auc_score:.2%}")
        print(f"AUC(2): {auc_score_2:.2%}")



if __name__ == '__main__':
    for i in range(10):
        print(f"========================")
        main(np.linspace(0.02, 0.04, 3))
