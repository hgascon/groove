#!/usr/bin/python
# groove - a graph tool for vector embeddings
# Hugo Gascon <hgascon@mail.de>


from scipy.sparse import coo_matrix
from collections import Counter
from simhash import Simhash
import numpy as np


class NeighborhoodHasher():
    """
    :param property: node property to compute hash
    :param f: bit length of the simhash
    """

    def __init__(self, hash_type="simhash",
                 max_node_size=0, hashing_property="label",
                 min_ngram_size=2, max_ngram_size=2, f=36):

        self.hash_type = hash_type
        self.max_node_size = max_node_size
        self.hashing_property = hashing_property
        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size
        self.hash_function_bit_length = f

    def fit(self, g):
        """ Embed a graph in a vector space using the neighborhood hash,
        where dimension represent the counts of an individual hash.
        :param g: a networkx graph object
        :return: sparse embedded vector csr_matrix
        """
        n = g.number_of_nodes()
        if n < self.max_node_size or self.max_node_size == 0:
            if n > 0:
                if self.hash_type == "simhash":
                    x = self._simhash_histogram(g)
                elif self.hash_type == "lineargk":
                    x = self._lineargk_histogram(g)
        else:
            raise("Number of nodes in graph exceeded!")
        return x

    def _lineargk_histogram(self, g):
        """ Compute the neighborhood hash of a graph g and return
            the histogram of the hashed labels.
        """
        gc = g.copy()
        label = self.hashing_property
        for node in iter(g.nodes()):
            neighbors_labels = [g.node[n][label]
                                for n in g.neighbors_iter(node)]
            if len(neighbors_labels) > 0:
                x = neighbors_labels[0]
                for i in neighbors_labels[1:]:
                    x = np.bitwise_xor(x, i)
                node_label = g.node[node][label]
                nh = np.bitwise_xor(np.roll(node_label, 1), x)
            else:
                nh = g.node[node][label]
            gc.node[node][label] = nh
        hashes = [g.node[name][label] for name in gc.nodes()]

        return self._vectorize(hashes, f=len(hashes[0]))

    def _simhash_histogram(self, g):

        """ At this time, scipy does not suuport 64-bit indexing
        so 63 bits is the max possible value for the hash function.

        :param g: networkx graph
        :return: scipy coo sparse matrix
        """
        hashes = []
        f = self.hash_function_bit_length
        #TODO consider the case of nodes without neighbors
        for node in iter(g.nodes()):
            neighbors_features = [g.node[n][self.hashing_property] for n
                                  in g.neighbors_iter(node)]
            features = []
            for nf in neighbors_features:
                features += [i for i in _ngrams(nf,
                                                self.min_ngram_size,
                                                self.max_ngram_size)]
            if neighbors_features:
                hashes += [Simhash(features,
                                   f=f).value]
        return self._vectorize(hashes, f)
    
    def _vectorize(self, hashes, f):
        c = Counter(hashes)
        data = np.array(c.values())
        j = np.array(c.keys(), dtype=np.int64)
        i = np.zeros(len(j))
        x = coo_matrix((data, (i, j)), shape=(1, 2**(f+1)-1))
        return x



def _ngrams(tokens, min_n=2, max_n=2):
    """
    Find all ngrams within a series of tokens

    :param tokens: a list of strings
    """
    n_tokens = len(tokens)
    for i in xrange(n_tokens):
        for j in xrange(i+min_n, min(n_tokens, i+max_n)+1):
            yield tokens[i:j]


def _get_ngrams_from_str_list(tokens, delimiter='', post=True):
    if delimiter:
        i = 0
        if post:
            i = 1
        features = list(_ngrams(''.join([t.split(delimiter)[i]
                                for t in tokens])))
    else:
        features = list(_ngrams(' '.join(tokens)))

    for i, f in enumerate(features):
        try:
            features[i] = f.decode('utf-8')
        except UnicodeDecodeError:
            print "Unknown encoding found! ({}) " \
                  "[character ignored]".format(f)
            features[i] = unicode(f, errors='ignore')
        except UnicodeEncodeError:
            print "UnicodeEncodeError: {}".format(f)
            features[i] = f
    return features
