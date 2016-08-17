import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm
from matrix import add_row_padding
from networkx.drawing.nx_agraph import read_dot
from graph import sort_nodes, unpack_str_vector_label, neighborhood_indexes


class NGFGenerator(object):

    def __init__(self, rng, data_size, vect_size, n_fingerprint):


        # HIDDEN LAYERS
        H1 = np.asarray(rng.uniform(low=-np.sqrt(6. / (2 * vect_size)),
                                    high=np.sqrt(6. / (2 * vect_size)),
                                    size=(vect_size, vect_size)),
                        dtype=theano.config.floatX)
        self.H1 = theano.shared(value=H1, name='H1', borrow=True)

        H2 = np.asarray(rng.uniform(low=-np.sqrt(6. / (2 * vect_size)),
                                    high=np.sqrt(6. / (2 * vect_size)),
                                    size=(vect_size, vect_size)),
                        dtype=theano.config.floatX)
        self.H2 = theano.shared(value=H2, name='H2', borrow=True)

        # OUTPUT WEIGHTS
        W1 = np.asarray(rng.uniform(low=-np.sqrt(6. / (vect_size + n_fingerprint)),
                                    high=np.sqrt(6. / (vect_size + n_fingerprint)),
                                    size=(vect_size, n_fingerprint)),
                        dtype=theano.config.floatX)
        self.W1 = theano.shared(value=W1, name='W1', borrow=True)

        W2 = np.asarray(rng.uniform(low=-np.sqrt(6. / (vect_size + n_fingerprint)),
                                    high=np.sqrt(6. / (vect_size + n_fingerprint)),
                                    size=(vect_size, n_fingerprint)),
                        dtype=theano.config.floatX)
        self.W2 = theano.shared(value=W2, name='W2', borrow=True)

        self.G = T.ftensor3('G') # mini-batch graph tensor
        self.I0 = T.itensor3('I0') # mini-batch neighborhood indexes 0
        self.I1 = T.itensor3('I1') # mini-batch neighborhood indexes 1

    def build_F(self):

        # L1
        G_n = self.G[self.I0, self.I1]
        r_a = T.nnet.nnet.sigmoid(T.sum(T.dot(G_n, self.H1), axis=2))
        dot = T.tensordot(r_a, self.W1, axes=1)
        F1 = T.theano.scan(fn=lambda d: T.sum(T.nnet.nnet.softmax(d), axis=0),
                           sequences=dot)[0]

        # prepare ra for indexing with the same dim as self.G
        zeros = T.zeros_like(self.G)
        r_a = T.set_subtensor(zeros[:, 1:, :], r_a)

        # L2
        G_n = r_a[self.I0, self.I1]
        r_a = T.nnet.nnet.sigmoid(T.sum(T.dot(G_n, self.H2), axis=2))
        dot = T.tensordot(r_a, self.W2, axes=1)
        F2 = T.theano.scan(fn=lambda d: T.sum(T.nnet.nnet.softmax(d), axis=0),
                           sequences=dot)[0]
        F = T.add(F1, F2)

        return F

    def build_training_function(self, data, batch_size):

        graphs, i0, i1 = data
        index = T.lscalar('index')
        print "> Defining output"
        out = self.build_F()
        print "> Compiling NGF function"
        train_fn = theano.function(
            inputs=[index],
            outputs=[out],
            allow_input_downcast=True,
            on_unused_input='ignore',
            givens={self.G: graphs[index * batch_size: (index + 1) * batch_size],
                    self.I0: i0,
                    self.I1: i1[index * batch_size: (index + 1) * batch_size]})
        print "> NGF function compiled"
        return train_fn


class NGF:

    def __init__(self):
        # default neighborhood of depth 2 (root + neighs)
        self.G = []
        self.I0 = []
        self.I1 = []
        self.F = []

    def load_data(self, data_path, n_depth=2, node_label='label'):

        files = []
        print "[+] Reading dot files"
        for f in os.listdir(data_path):
            if f.endswith('dot'):
                files.append(os.path.abspath(os.path.join(data_path, f)))

        G, I = [], []
        assert files

        print "[+] Computing tensor"
        for i in tqdm(range(len(files))):
            f = files[i]
            graph = read_dot(f)
            G_g = []
            try:
                nodes = sort_nodes(graph)
            except ValueError:
                tqdm.write("> Ignoring graph without nodes {}".format(f))
                continue
            # iterate over neighborhoods sorted by the distance of
            # their central node to root and build corresponding tensor
            for node in nodes:
                n_label = graph.node[node][node_label]
                n_label = unpack_str_vector_label(n_label)
                G_g.append(n_label)
            G.append(G_g)
            indexes = neighborhood_indexes(graph, nodes, node_label)
            I.append(indexes)

        print "[+] Formatting tensor"
        G = np.array([np.array(x) for x in G])

        # remove graphs without enough neighborhoods
        empty_graphs_idx = [i for i, idx in enumerate(G) if len(idx) <= 1]
        G = np.delete(G, empty_graphs_idx)
        I = np.delete(np.array(I), empty_graphs_idx)

        # we add zero padding to all graphs
        # up to the size of the largest graph
        cols = G[0].shape[1]
        # we add a zero row on the top of each graph to index wih zeros
        G = np.array([np.vstack([np.zeros(cols), rows]) for rows in G])
        G = add_row_padding(G)
        G = np.array(G, dtype=theano.config.floatX)
        G = theano.shared(G, name='G')

        # find max dimensions in list of neighborhood indexes
        max_rows = 0
        for g_indexes in I:
            max_rows = max([max_rows, len(g_indexes)])
        max_cols = max_rows

        # we build the first index
        o = np.ones(max_rows, dtype=np.int)
        i0 = np.array([[i * o] for i in np.arange(self.batch_size)])
        i0 = theano.shared(np.array(i0, dtype=np.int32), name='i0')

        # add zero padding to lists of indexes
        i1 = []
        for g_indexes in I:
            g_indexes_padded = []
            for nn_g_indexes in g_indexes:
                g_indexes_padded.append(np.hstack([
                    np.array(nn_g_indexes, dtype=np.int32) + 1,
                    np.zeros((max_cols - len(nn_g_indexes)))]))
            g_indexes_padded = np.array(g_indexes_padded)
            i1.append(np.vstack([
                g_indexes_padded,
                np.zeros((max_rows - g_indexes_padded.shape[0], max_cols))]))
        i1 = theano.shared(np.array(i1, dtype=np.int32), name='i1')

        print "G", G.get_value().shape
        print "i0", i0.get_value().shape
        print "i1", i1.get_value().shape
        self.G = G
        self.I0 = i0
        self.I1 = i1

    def fit(self, data_path, batch_size=5, n_fingerprint=64):

        graphs = self.G
        data_shape = graphs.get_value(borrow=True).shape
        n_train_batches = data_shape[0] // batch_size
        data_size = data_shape[0]
        vect_size = data_shape[2]
        rng = np.random.RandomState(123)

        # TODO assert that all graphs have more nodes than n_hid
        ngf = NGFGenerator(rng=rng, data_size=data_size, vect_size=vect_size,
                           n_fingerprint=n_fingerprint)
        start_time = timeit.default_timer()
        data = self.G, self.I0, self.I1
        train_fn = ngf.build_training_function(data, batch_size)
        end_time = timeit.default_timer()
        compile_time = (end_time - start_time)
        print >> sys.stderr, ('The function compiled in %.2fm' % (compile_time / 60.))

        print "> Starting embedding\n\n[batch size {} | " \
              "{} batches]\n".format(batch_size, n_train_batches)
        start_time = timeit.default_timer()
        for batch_index in xrange(n_train_batches):
            self.F.append(train_fn(batch_index))
        end_time = timeit.default_timer()
        embedding_time = (end_time - start_time)
        print >> sys.stderr, ('The code ran for %.2fm' % (embedding_time / 60.))
