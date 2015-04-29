#!/usr/bin/python
# groove - a graph tool for vector embeddings
# Hugo Gascon <hgascon@mail.de>


import numpy as np
from scipy.sparse import csr_matrix

MAX_NODE_SIZE = 0

def embed(g):
    """ Embed a graph in a vector space using the neighborhood hash,
    where dimension represent the counts of an individual hash.
    :param g: a networkx graph object
    :return: sparse embedded vector csr_matrix
    """
    n = g.number_of_nodes()
    if n < MAX_NODE_SIZE or MAX_NODE_SIZE == 0:
        if n > 0:
            x = csr_matrix(compute_label_histogram(g))
    else:
        raise("Number of nodes in graph exceeded!")
    return x


def compute_label_histogram(g):
    """ Compute the neighborhood hash of a graph g and return
        the histogram of the hashed labels.
    """
    g_hash = neighborhood_hash(g)
    g_x = label_histogram(g_hash)
    return g_x
   
   
def neighborhood_hash(g):
    """ Compute the simple neighborhood hashed version of a graph.
    """
    gnh = g.copy()
    for node in iter(g.nodes()):
        neighbors_labels = [g.node[n]["label"] for n in g.neighbors_iter(node)]
        if len(neighbors_labels) > 0:
            x = neighbors_labels[0]
            for i in neighbors_labels[1:]:
                x = np.bitwise_xor(x, i)
            node_label = g.node[node]["label"]
            nh = np.bitwise_xor(np.roll(node_label, 1), x)
        else:
            nh = g.node[node]["label"]

        gnh.node[node]["label"] = nh

    return gnh


def label_histogram(g):
    """ Compute the histogram of labels in nx graph g. Every label is a
        binary array. The histogram length is 2**len(label)
    """
    labels = [g.node[name]["label"] for name in g.nodes()]
    h = np.zeros(2 ** len(labels[0]))
    for l in labels:
        h[int(''.join([str(i) for i in l]), base=2)] += 1
    return h
