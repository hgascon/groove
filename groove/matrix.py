import numpy as np
from scipy.sparse import csr_matrix

""" This module implements a series of operations over
    numpy and scipy CSR matrices
"""


def csr_vappend(a, b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of
    the first one.  Much faster than scipy.sparse.vstack but assumes the
    type to be csr and overwrites the first matrix instead of copying it.
    The data, indices, and indptr still get copied.

    :param a: csr matrix
    :param b: csr matrix
    :return: csr matrix (a is overwritten)
    """

    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])


def csr_happend(a, b):
    """ Takes in 2 csr_matrices and appends the second one to the right of
    the first one.  Much faster than scipy.sparse.hstack but assumes the
    type to be csr and overwrites the first matrix instead of copying it.
    The data, indices, and indptr still get copied.

    :param a: csr matrix
    :param b: csr matrix
    :return: csr matrix (a is overwritten)
    """

    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices + a.shape[1]))
    a.indptr[1] += b.indptr[1]
    a._shape = (a.shape[0], a.shape[1] + b.shape[1])


def add_matrices_padding(a, max_rows=None, max_matrices=None):
    """
    Given a N list of arrays (m, n, d) with different m and n
    but same dimensionality d, add zero padding up to the
    maximum of m and n values and return an array (N, M, N, d).

    :param a: array (N, m, n, d) with variable m and n
    :return: array (N, M, N, d) with M=max(all m) and N=max(all n)
    """
    if not max_rows:
        max_rows = max([a_i.shape[0] for a_i in a])
    a_padded = []
    cols = a[0].shape[1]
    for a_i in a:
        rows = a_i.shape[0]
        a_padded.append(np.vstack((a_i, np.zeros((max_rows - rows, cols)))))
    if max_matrices:
        for i in range(max_matrices - len(a_padded)):
            a_padded.append(np.zeros((max_rows, cols)))

    return np.array(a_padded, dtype=np.float32)


def add_row_padding(a, max_rows=None):
    """
    Given a list of numpy matrices with different number
    of rows, but same dimensionality, add zero padding to
    all matrices up to the size of the largest matrix.

    :param a: numpy array with dimensions (N, m, n) with variable m
    :return: numpy array with dimensions (N, M, n) with M=max(all m)
    """
    if not max_rows:
        max_rows = max([a_i.shape[0] for a_i in a])
    a_padded = []
    cols = a[0].shape[1]
    for a_i in a:
        rows = a_i.shape[0]
        a_padded.append(np.vstack((a_i, np.zeros((max_rows - rows, cols)))))

    return np.array(a_padded, dtype=np.float32)


def add_list_csr_matrices_padding(a, max_rows=None, max_matrices=None):
    """
    Given a list of CSR matrices with different number
    of rows, but same dimensionality, add zero padding to
    all matrices up to the size of the largest matrix.

    :param a: list of CSR matrices
    :return: list of zero-padded numpy arrays
    """
    if not max_rows:
        max_rows = max([a_i.shape[0] for a_i in a])
    a_padded = []

    cols = a[0].shape[1]
    for a_i in a:
        rows = a_i.shape[0]
        a_padded.append(np.vstack((a_i.todense(),
                                   np.zeros((max_rows - rows, cols)))))
    if max_matrices:
        for i in range(max_matrices - len(a_padded)):
            a_padded.append(np.zeros((max_rows, cols)))

    return a_padded


def add_csr_matrices_padding(a):
    """
    Given a matrix of dimension 3, add zero padding to
    all matrices of dimension 2 up to the size of the
    largest matrix.

    :param a: csr matrix
    :return: list of padded numpy arrays
    """
    max_rows = max([a_i.shape[0] for a_i in a])
    a_padded = []
    for a_i in a:
        rows, cols = a_i.shape
        csr_vappend(a_i, csr_matrix(np.zeros((max_rows - rows, cols))))
        a_padded.append(a_i)
    return a_padded
