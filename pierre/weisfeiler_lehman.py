"""The weisfeiler lehman kernel :cite:`shervashidze2011weisfeiler`."""
# Author: Ioannis Siglidis <y.siglidis@gmail.com>
# License: BSD 3 clause
import warnings
from typing import List
import networkx
import sys

sys.path.append('.')
import numpy as np
from .vertex_histogram import VertexHistogram

from six import iteritems
from six import itervalues
from six.moves.collections_abc import Iterable


class WeisfeilerLehman():
    """Compute the Weisfeiler Lehman Kernel."""

    def __init__(self, n_jobs=None, verbose=False,
                 normalize=False, n_iter=5, base_graph_kernel=VertexHistogram):
        """Initialise a `weisfeiler_lehman` kernel."""

        self.n_iter = n_iter
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.params = dict()
        self.params["verbose"] = verbose
        self.params["n_jobs"] = n_jobs
        self.params['normalize'] = normalize

    def parse_input(self, X: List[networkx.Graph]):
        """Parse input for weisfeiler lehman."""

        # Input validation and parsing
        nx = 0
        Gs_ed, L, distinct_values, extras = dict(), dict(), set(), dict()
        for (idx, x) in enumerate(X):
            Gs_ed[nx] = {
                k: {
                    v: 1.0 for v in x.nodes if x.get_edge_data(k, v) is not None
                } for k in x.nodes
            }
            L[nx] = {k: x.nodes[k]['labels'][0] for k in x.nodes}
            extras[nx] = ({ed: x.get_edge_data(ed[0], ed[1])['labels'][0] for ed in x.edges},) + tuple()
            distinct_values = np.unique(list(L[nx].values()) + list(distinct_values))
            nx += 1

        # get all the distinct values of current labels
        WL_labels_inverse = dict()

        # assign a number to each label
        label_count = 0
        for dv in sorted(list(distinct_values)):
            WL_labels_inverse[dv] = label_count
            label_count += 1

        # Initalize an inverse dictionary of labels for all iterations
        self._inv_labels = dict()
        self._inv_labels[0] = WL_labels_inverse

        def generate_graphs(label_count, WL_labels_inverse):
            new_graphs = list()
            for j in range(nx):
                new_labels = dict()
                for k in L[j].keys():
                    new_labels[k] = WL_labels_inverse[L[j][k]]
                L[j] = new_labels
                # add new labels
                new_graphs.append((Gs_ed[j], new_labels) + extras[j])
            yield new_graphs

            for i in range(1, self.n_iter):
                label_set, WL_labels_inverse, L_temp = set(), dict(), dict()
                for j in range(nx):
                    # Find unique labels and sort
                    # them for both graphs
                    # Keep for each node the temporary
                    L_temp[j] = dict()
                    for v in Gs_ed[j].keys():
                        credential = str(L[j][v]) + "," + \
                                     str(sorted([L[j][n] for n in Gs_ed[j][v].keys()]))
                        L_temp[j][v] = credential
                        label_set.add(credential)

                label_list = sorted(list(label_set))
                for dv in label_list:
                    WL_labels_inverse[dv] = label_count
                    label_count += 1

                # Recalculate labels
                new_graphs = list()
                for j in range(nx):
                    new_labels = dict()
                    for k in L_temp[j].keys():
                        new_labels[k] = WL_labels_inverse[L_temp[j][k]]
                    L[j] = new_labels
                    # relabel
                    new_graphs.append((Gs_ed[j], new_labels) + extras[j])
                self._inv_labels[i] = WL_labels_inverse
                yield new_graphs

        base_graph_kernel = {i: self.base_graph_kernel(**self.params) for i in range(self.n_iter)}
        if self._method_calling == 1:
            for (i, g) in enumerate(generate_graphs(label_count, WL_labels_inverse)):
                base_graph_kernel[i].fit(g)
        elif self._method_calling == 2:
            graphs = generate_graphs(label_count, WL_labels_inverse)
            values = [
                base_graph_kernel[i].fit_transform(g) for (i, g) in enumerate(graphs)
            ]
            K = np.sum(values, axis=0)

        if self._method_calling == 1:
            return base_graph_kernel
        elif self._method_calling == 2:
            return K, base_graph_kernel

    def fit_transform(self, X, y=None):
        """Fit and transform, on the same dataset.

        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). If None the kernel matrix is calculated upon fit data.
            The test samples.

        y : Object, default=None
            Ignored argument, added for the pipeline.

        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target and features

        """
        self._method_calling = 2
        if X is None:
            raise ValueError('transform input cannot be None')
        else:
            km, self.X = self.parse_input(X)

        self._X_diag = np.diagonal(km)
        if self.normalize:
            old_settings = np.seterr(divide='ignore')
            km = np.nan_to_num(np.divide(km, np.sqrt(np.outer(self._X_diag, self._X_diag))))
            np.seterr(**old_settings)
        return km

    def transform(self, X):
        """Calculate the kernel matrix, between given and fitted dataset.

        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). If None the kernel matrix is calculated upon fit data.
            The test samples.

        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features

        """
        self._method_calling = 3

        # Input validation and parsing
        if X is None:
            raise ValueError('transform input cannot be None')
        else:
            nx = 0
            distinct_values = set()
            Gs_ed, L = dict(), dict()
            for (i, x) in enumerate(iter(X)):
                Gs_ed[nx] = {
                    k: {
                        v: 1.0 for v in x.nodes if x.get_edge_data(k, v) is not None
                    } for k in x.nodes
                }
                L[nx] = {k: x.nodes[k]['labels'][0] for k in x.nodes}

                # Hold all the distinct values
                distinct_values |= set(
                    v for v in itervalues(L[nx])
                    if v not in self._inv_labels[0])
                nx += 1
            if nx == 0:
                raise ValueError('parsed input is empty')

        nl = len(self._inv_labels[0])
        WL_labels_inverse = {dv: idx for (idx, dv) in
                             enumerate(sorted(list(distinct_values)), nl)}

        def generate_graphs(WL_labels_inverse, nl):
            # calculate the kernel matrix for the 0 iteration
            new_graphs = list()
            for j in range(nx):
                new_labels = dict()
                for (k, v) in iteritems(L[j]):
                    if v in self._inv_labels[0]:
                        new_labels[k] = self._inv_labels[0][v]
                    else:
                        new_labels[k] = WL_labels_inverse[v]
                L[j] = new_labels
                # produce the new graphs
                new_graphs.append([Gs_ed[j], new_labels])
            yield new_graphs

            for i in range(1, self.n_iter):
                new_graphs = list()
                L_temp, label_set = dict(), set()
                nl += len(self._inv_labels[i])
                for j in range(nx):
                    # Find unique labels and sort them for both graphs
                    # Keep for each node the temporary
                    L_temp[j] = dict()
                    for v in Gs_ed[j].keys():
                        credential = str(L[j][v]) + "," + \
                                     str(sorted([L[j][n] for n in Gs_ed[j][v].keys()]))
                        L_temp[j][v] = credential
                        if credential not in self._inv_labels[i]:
                            label_set.add(credential)

                # Calculate the new label_set
                WL_labels_inverse = dict()
                if len(label_set) > 0:
                    for dv in sorted(list(label_set)):
                        idx = len(WL_labels_inverse) + nl
                        WL_labels_inverse[dv] = idx

                # Recalculate labels
                new_graphs = list()
                for j in range(nx):
                    new_labels = dict()
                    for (k, v) in iteritems(L_temp[j]):
                        if v in self._inv_labels[i]:
                            new_labels[k] = self._inv_labels[i][v]
                        else:
                            new_labels[k] = WL_labels_inverse[v]
                    L[j] = new_labels
                    # Create the new graphs with the new labels.
                    new_graphs.append([Gs_ed[j], new_labels])
                yield new_graphs

        # Calculate the kernel matrix without parallelization
        graphs = generate_graphs(WL_labels_inverse, nl)
        values = [self.X[i].transform(g) for (i, g) in enumerate(graphs)]
        K = np.sum(values, axis=0)

        self._is_transformed = True
        if self.normalize:
            X_diag, Y_diag = self.diagonal()
            old_settings = np.seterr(divide='ignore')
            K = np.nan_to_num(np.divide(K, np.sqrt(np.outer(Y_diag, X_diag))))
            np.seterr(**old_settings)

        return K

    def diagonal(self):
        """Calculate the kernel matrix diagonal for fitted data.

        A funtion called on transform on a seperate dataset to apply
        normalization on the exterior.

        Parameters
        ----------
        None.

        Returns
        -------
        X_diag : np.array
            The diagonal of the kernel matrix, of the fitted data.
            This consists of kernel calculation for each element with itself.

        Y_diag : np.array
            The diagonal of the kernel matrix, of the transformed data.
            This consists of kernel calculation for each element with itself.

        """
        # Calculate diagonal of X
        Y_diag = None
        if self._is_transformed:
            X_diag, Y_diag = self.X[0].diagonal()
            # X_diag is considered a mutable and should not affect the kernel matrix itself.
            X_diag.flags.writeable = True
            for i in range(1, self.n_iter):
                x, y = self.X[i].diagonal()
                X_diag += x
                Y_diag += y
            self._X_diag = X_diag
        else:
            # case sub kernel is only fitted
            X_diag = self.X[0].diagonal()
            # X_diag is considered a mutable and should not affect the kernel matrix itself.
            X_diag.flags.writeable = True
            for i in range(1, self.n_iter):
                x = self.X[i].diagonal()
                X_diag += x
            self._X_diag = X_diag

        if self._is_transformed:
            return self._X_diag, Y_diag
        else:
            return self._X_diag


def efit(object, data):
    """Fit an object on data."""
    object.fit(data)


def efit_transform(object, data):
    """Fit-Transform an object on data."""
    return object.fit_transform(data)


def etransform(object, data):
    """Transform an object on data."""
    return object.transform(data)
