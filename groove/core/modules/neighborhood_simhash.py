# groove - a graph tool for vector embeddings
# Hugo Gascon <hgascon@mail.de>

import numpy as np
from scipy.sparse import coo_matrix
from simhash import Simhash
from collections import Counter

from groove.core.util import ngrams


MAX_NODE_SIZE = 0
NGRAM_SIZE = 2

def embed(g):
    """ Embed a graph in a vector space using the neighborhood hash,
    where dimension represent the counts of an individual hash.
    :param g: a networkx graph object
    :return: a numpy array x
    """
    n = g.number_of_nodes()
    if n < MAX_NODE_SIZE or MAX_NODE_SIZE == 0:
        if n > 0:
            x = compute_simhash_histogram(g)
    else:
        raise("Number of nodes in graph exceeded!")
    return x

def compute_simhash_histogram(g, property="label", f=63):

    """ At this time, scipy does not suuport 64-bit indexing
    so 63 bits is the max possible value for the hash function.

    :param g: networkx graph
    :param property: node property to compute hash
    :param f: bit length of the simhash
    :return: scipy coo sparse matrix
    """
    hashes = []
    for node in iter(g.nodes()):
        neighbors_features = [g.node[n][property] for n
                              in g.neighbors_iter(node)]
        features = []
        for nf in neighbors_features:
            features += [i for i in ngrams(nf, NGRAM_SIZE)]
            
        if neighbors_features:
            hashes += [Simhash(features, f=f).value]

    c = Counter(hashes)
    data = np.array(c.values())
    j = np.array(c.keys(), dtype=np.int64)

    i = np.zeros(len(j))
    x = coo_matrix((data, (i, j)), shape=(1, 2**(f+1)-1))
    return x

