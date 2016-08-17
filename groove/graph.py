import numpy as np
import networkx as nx
from bitarray import bitarray
from networkx.exception import NetworkXNoPath

""" This module implements a series of operations over NX graphs
"""


def sort_neighbors_bin(g, neighbors, node_label):
    vectors = np.array([])
    for n in neighbors:
        b = bitarray()
        b.extend((g.node[n][node_label].todense() > 0).tolist()[0])
        vectors = np.hstack((vectors, int(b.to01(), 2)))
    return list(np.array(neighbors)[vectors.argsort()])


def sort_neighbors(g, neighbors, node_label):
    vectors = np.array([])
    for n in neighbors:
        b = bitarray()
        label = unpack_str_vector_label(g.node[n][node_label])
        label_bin = (label > 0).astype(int)
        b.extend(label_bin.tolist())
        vectors = np.hstack((vectors, int(b.to01(), 2)))
    return list(np.array(neighbors)[vectors.argsort()])


def select_root(g):
    roots = [n for n, d in g.in_degree().items() if d == 0]
    if roots:
        root = np.random.choice(roots)
    else:
        root = np.random.choice(g.nodes())
    # print "Randomly selecting root {} from nodes {}".format(root, roots)
    return root


def read_entry_point(g):
    try:
        return g.graph['graph']['entry_point']
    except KeyError:
        return select_root(g)


def sort_nodes(g):
    nodes = g.nodes()
    if not nodes:
        raise ValueError("Graph {} has no nodes!".format(g))
    root = read_entry_point(g)
    distances = np.array([])
    for n in nodes:
        try:
            d = len(nx.shortest_path(g, root, n)) - 1
        except NetworkXNoPath:
            d = -1
        distances = np.hstack([distances, d])
    distances[np.where(distances == -1)] = distances.max() + 1
    return list(np.array(nodes)[np.argsort(distances)])


def unpack_str_vector_label(l):
    return np.array([int(i) for i in l.split(';')])


def neighborhood_indexes(g, nodes, node_label):
    nbs = [[n] + sort_neighbors(g, nx.neighbors(g, n), node_label)
           for n in nodes]
    indexes = []
    for nb in nbs:
        indexes.append([nodes.index(n) for n in nb])
    return np.array(indexes)


def remove_isolates(g):
    # remove isolated nodes
    g.remove_nodes_from(nx.algorithms.isolates(g))
    return g


def remove_non_reachables(g):
    # remove nodes not reachable from the root
    all_nodes = set(g.nodes())
    try:
        entry_point = g.graph['graph']['entry_point']
        all_nodes.remove(entry_point)
        g.remove_nodes_from(all_nodes ^ nx.descendants(g, entry_point))
    except:
        print "Error loading entry point from graph. Graph not modified."
    return g


def get_neighborhood(g, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(g, node)
    return [nnode for nnode, length in path_lengths.iteritems()
            if length in np.arange(n) + 1]
