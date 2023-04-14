from typing import List

import networkx as nx
import numpy as np
from grakel import NeighborhoodHash, Graph
from six import itervalues, iteritems
NeighborhoodHash



class NeighborhoodHashKernel:
    def __init__(self, R, random_state):
        self.R = R
        self._label_count = 0

        self._bit_count = 0
        self._mask = 0
        self._random_state = random_state
    def fit(self, X: List[nx.Graph]):
        self._method_calling = 1

        i = 0
        out = list()
        gs = list()
        self._labels_hash_dict, labels_hash_set = dict(), set()
        for (idx, x) in enumerate(X):
            # x = Graph(x[0], x[1], {}, self._graph_format)
            vertices = x.nodes
            labels = nx.get_node_attributes(x, "labels")

            g = (vertices, labels,
                 {n: list(x.neighbors(n)) for n in x.nodes})

            # collect all the labels
            labels_hash_set |= set(itervalues(labels))
            gs.append(g)
            i += 1

        self._label_count = len(labels_hash_set)
        self._bit_count = 1 << self._label_count
        self._mask = self._label_count - 1

        nl = self._random_state.choice(self._label_count, len(labels_hash_set),
                              replace=False).tolist()

        self._labels_hash_dict = dict(zip(labels_hash_set, nl))

        # for all graphs
        for vertices, labels, neighbors in gs:
            new_labels = {v: self._labels_hash_dict[l]
                          for v, l in iteritems(labels)}
            g = (vertices, new_labels, neighbors,)
            gr = {0: self.neighborhood_hash_count_sensitive(g)}
            for r in range(1, self.R):
                gr[r] = self.neighborhood_hash_count_sensitive(gr[r - 1])

            # save the output for all levels
            out.append(gr)

        self.X = out

        # Return the transformer
        return self

    def neighborhood_hash_count_sensitive(self, G):
        """Count sensitive neighborhood hash as defined in :cite:`Hido2009ALG`.

        Parameters
        ----------
        G : tuple, len=3
           A tuple three elements consisting of vertices sorted by labels,
           vertex label dict, edge dict and number of occurencies dict for
           labels.

        Returns
        -------
        vertices_labels_edges_noc : tuple
            A tuple of vertices, new_labels-dictionary and edges.

        """
        vertices, labels, neighbors = G
        new_labels = dict()
        for u in vertices:
            if (labels[u] is None or
                    any(labels[n] is None for n in neighbors[u])):
                new_labels[u] = None
            else:
                label = self.ROT(labels[u], 1)
                label ^= self.radix_sort_rot([labels[n] for n in neighbors[u]])
                new_labels[u] = label

        return tuple(self._vertex_sort(vertices, new_labels)) + (neighbors,)

    def ROT(self, n, d):
        """`rot` operation for binary numbers.

        Parameters
        ----------
        n : int
            The value which will be rotated.

        d : int
            The number of rotations.

        Returns
        -------
        rot : int
            The result of a rot operation.

        """
        m = d % self._bit_count

        if m > 0:
            return (n << m) & self._mask | \
                ((n & self._mask) >> (self._bit_count - m))
        else:
            return n

    def radix_sort_rot(self, labels):
        """Sorts vertices based on labels.

        Parameters
        ----------
        labels : dict
            Dictionary of labels for vertices.

        Returns
        -------
        labels_counts : list
            A list of labels with their counts (sorted).

        """
        n = len(labels)
        result = 0
        if n == 0:
            return result

        for b in range(self._bit_count):
            # The output array elements that will have sorted arr
            output = [0] * n

            # initialize count array as 0
            count = [0, 0]

            # Store count of occurrences in count[]
            for i in range(n):
                count[(labels[i] >> b) % 2] += 1

            # Change count[i] so that count[i] now contains actual
            #  position of this digit in output array
            count[1] += count[0]

            # Build the output array
            for i in range(n - 1, -1, -1):
                index = (labels[i] >> b)
                output[count[index % 2] - 1] = labels[i]
                count[index % 2] -= 1

            # Copying the output array to arr[],
            # so that arr now contains sorted numbers
            labels = output

        previous, occ = labels[0], 1
        for i in range(1, len(labels)):
            label = labels[i]
            if label == previous:
                occ += 1
            else:
                result ^= self.ROT(previous ^ occ, occ)
                occ = 1
            previous = label
        if occ > 0:
            result ^= self.ROT(previous ^ occ, occ)
        return result

    def _vertex_sort(self, vertices, labels):
        """Sorts vertices based on labels.

        Parameters
        ----------
        vertices : listable
            A listable of vertices.

        labels : dict
            Dictionary of labels for vertices.

        Returns
        -------
        vertices_labels : tuple, len=2
            The sorted vertices based on labels and labels for vertices.

        """
        if self._method_calling == 3:
            return (sorted(list(vertices),
                           key=lambda x: float('inf')
                           if labels[x] is None else labels[x]), labels)
        else:
            return (sorted(vertices, key=lambda x: labels[x]), labels)
