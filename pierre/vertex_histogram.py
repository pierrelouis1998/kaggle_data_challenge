"""The vertex kernel as defined in :cite:`sugiyama2015halting`."""
# Author: Ioannis Siglidis <y.siglidis@gmail.com>
# License: BSD 3 clause
from warnings import warn

from collections import Counter

import numpy as np
from numpy import zeros
from numpy import einsum
from numpy import array
from numpy import squeeze
from scipy.sparse import csr_matrix

# Python 2/3 cross-compatibility import
from six import iteritems
from six import itervalues
from six.moves.collections_abc import Iterable


class VertexHistogram():
    """Vertex Histogram kernel as found in :cite:`sugiyama2015halting`.
    """

    def __init__(self, n_jobs=None, normalize=False, verbose=False, sparse='auto'):
        """Initialise a vertex histogram kernel."""
        self.sparse = sparse
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.normalize = normalize
        self._initialized = dict(n_jobs=False)

    def fit(self, X, y=None):
        """Fit a dataset, for a transformer.

        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). The train samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
        Returns self.

        """
        self._is_transformed = False
        self._method_calling = 1

        # Input validation and parsing
        if X is None:
            raise ValueError('`fit` input cannot be None')
        else:
            self.X = self.parse_input(X)

        # Return the transformer
        return self

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
            raise ValueError('`transform` input cannot be None')
        else:
            Y = self.parse_input(X)

        # Transform - calculate kernel matrix
        km = self._calculate_kernel_matrix(Y)
        self._Y = Y

        # Self transform must appear before the diagonal call on normilization
        self._is_transformed = True
        if self.normalize:
            X_diag, Y_diag = self.diagonal()
            km /= np.sqrt(np.outer(Y_diag, X_diag))
        return km

    def fit_transform(self, X):
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

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features

        """
        self._method_calling = 2
        self.fit(X)

        # Transform - calculate kernel matrix
        km = self._calculate_kernel_matrix()

        self._X_diag = np.diagonal(km)
        if self.normalize:
            return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
        else:
            return km

    def parse_input(self, X):
        """Parse and check the given input for VH kernel.

        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format).

        Returns
        -------
        out : np.array, shape=(len(X), n_labels)
            A np.array for frequency (cols) histograms for all Graphs (rows).

        """
        if not isinstance(X, Iterable):
            raise TypeError('input must be an iterable\n')
        else:
            rows, cols, data = list(), list(), list()
            if self._method_calling in [1, 2]:
                labels = dict()
                self._labels = labels
            elif self._method_calling == 3:
                labels = dict(self._labels)
            ni = 0
            for (i, x) in enumerate(iter(X)):
                L = x[1]
                # construct the data input for the numpy array
                for (label, frequency) in iteritems(Counter(itervalues(L))):
                    # for the row that corresponds to that graph
                    rows.append(ni)

                    # and to the value that this label is indexed
                    col_idx = labels.get(label, None)
                    if col_idx is None:
                        # if not indexed, add the new index (the next)
                        col_idx = len(labels)
                        labels[label] = col_idx

                    # designate the certain column information
                    cols.append(col_idx)

                    # as well as the frequency value to data
                    data.append(frequency)
                ni += 1

            if self._method_calling in [1, 2]:
                if self.sparse == 'auto':
                    self.sparse_ = (len(cols)/float(ni * len(labels)) <= 0.5)
                else:
                    self.sparse_ = bool(self.sparse)

            if self.sparse_:
                features = csr_matrix((data, (rows, cols)), shape=(ni, len(labels)), copy=False)
            else:
                # Initialise the feature matrix
                try:
                    features = zeros(shape=(ni, len(labels)))
                    features[rows, cols] = data
                except MemoryError:
                    warn('memory-error: switching to sparse')
                    self.sparse_, features = True, csr_matrix((data, (rows, cols)), shape=(ni, len(labels)), copy=False)

            if ni == 0:
                raise ValueError('parsed input is empty')
            return features

    def _calculate_kernel_matrix(self, Y=None):
        """Calculate the kernel matrix given a target_graph and a kernel.

        Each a matrix is calculated between all elements of Y on the rows and
        all elements of X on the columns.

        Parameters
        ----------
        Y : np.array, default=None
            The array between samples and features.

        Returns
        -------
        K : numpy array, shape = [n_targets, n_inputs]
            The kernel matrix: a calculation between all pairs of graphs
            between targets and inputs. If Y is None targets and inputs
            are the taken from self.X. Otherwise Y corresponds to targets
            and self.X to inputs.

        """
        if Y is None:
            K = self.X.dot(self.X.T)
        else:
            K = Y[:, :self.X.shape[1]].dot(self.X.T)

        if self.sparse_:
            return K.toarray()
        else:
            return K

    def diagonal(self):
        """Calculate the kernel matrix diagonal of the fitted data.

        Parameters
        ----------
        None.

        Returns
        -------
        X_diag : np.array
            The diagonal of the kernel matrix, of the fitted. This consists
            of each element calculated with itself.


        """
        # Calculate diagonal of X
        if self.sparse_:
            self._X_diag = squeeze(array(self.X.multiply(self.X).sum(axis=1)))
        else:
            self._X_diag = einsum('ij,ij->i', self.X, self.X)
        if self.sparse_:
            Y_diag = squeeze(array(self._Y.multiply(self._Y).sum(axis=1)))
        else:
            Y_diag = einsum('ij,ij->i', self._Y, self._Y)
        return self._X_diag, Y_diag
