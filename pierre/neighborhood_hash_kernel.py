from grakel import NeighborhoodHash

NeighborhoodHash

class NeighborhoodHashKernel:
    def __init__(self, neighbor_count):
        self._neighbor_count = neighbor_count

    def fit(self, X):
        i = 0
        out = list()
        gs = list()
        self._labels_hash_dict, labels_hash_set = dict(), set()
        for (idx, x) in enumerate(iter(X)):
            is_iter = isinstance(x, Iterable)
            if is_iter:
                x = list(x)
            if is_iter and len(x) in [0, 1, 2, 3]:
                if len(x) == 0:
                    warnings.warn('Ignoring empty element on index: '
                                  + str(idx))
                    continue
                elif len(x) == 1:
                    warnings.warn(
                        'Ignoring empty element on index: '
                        + str(i) + '\nLabels must be provided.')
                else:
                    x = Graph(x[0], x[1], {}, self._graph_format)
                    vertices = list(x.get_vertices(purpose="any"))
                    Labels = x.get_labels(purpose="any")
            elif type(x) is Graph:
                vertices = list(x.get_vertices(purpose="any"))
                Labels = x.get_labels(purpose="any")
            else:
                raise TypeError('each element of X must be either '
                                'a graph object or a list with at '
                                'least a graph like object and '
                                'node labels dict \n')

            g = (vertices, Labels,
                 {n: x.neighbors(n, purpose="any") for n in vertices})

            # collect all the labels
            labels_hash_set |= set(itervalues(Labels))
            gs.append(g)
            i += 1

        if i == 0:
            raise ValueError('parsed input is empty')

        # Hash labels
        if len(labels_hash_set) > self._max_number:
            warnings.warn('Number of labels is smaller than'
                          'the biggest possible.. '
                          'Collisions will appear on the '
                          'new labels.')

            # If labels exceed the biggest possible size
            nl, nrl = list(), len(labels_hash_set)
            while nrl > self._max_number:
                nl += self.random_state_.choice(self._max_number,
                                                self._max_number,
                                                replace=False).tolist()
                nrl -= self._max_number
            if nrl > 0:
                nl += self.random_state_.choice(self._max_number,
                                                nrl,
                                                replace=False).tolist()
            # unify the collisions per element.

        else:
            # else draw n random numbers.
            nl = self.random_state_.choice(self._max_number, len(labels_hash_set),
                                           replace=False).tolist()

        self._labels_hash_dict = dict(zip(labels_hash_set, nl))

        # for all graphs
        for vertices, labels, neighbors in gs:
            new_labels = {v: self._labels_hash_dict[l]
                          for v, l in iteritems(labels)}
            g = (vertices, new_labels, neighbors,)
            gr = {0: self.NH_(g)}
            for r in range(1, self.R):
                gr[r] = self.NH_(gr[r - 1])

            # save the output for all levels
            out.append(gr)

        self.X = out

        # Return the transformer
        return self