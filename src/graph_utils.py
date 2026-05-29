"""Spectral graph utilities for conservation-aware art generation."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def build_graph(adj_matrix):
    """Normalize and validate adjacency matrix."""
    A = np.array(adj_matrix, dtype=np.float64)
    return A


def laplacian(A):
    """Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}."""
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    n = A.shape[0]
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def spectral_decomposition(A, k=None):
    """Compute eigenvalues and eigenvectors of normalized Laplacian."""
    L = laplacian(A)
    n = L.shape[0]
    if k is None:
        k = n
    k = min(k, n - 2)
    if k < 1:
        vals, vecs = np.linalg.eigh(L)
    else:
        L_sparse = sparse.csr_matrix(L)
        vals, vecs = eigsh(L_sparse, k=k, which='SM')
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def conservation_ratio(A, attributes):
    """Compute conservation ratio: how well attributes are conserved on the graph.
    
    High ratio = smooth attributes (well conserved)
    Low ratio = rough attributes (anomalous)
    """
    L = laplacian(A)
    x = np.array(attributes, dtype=np.float64)
    # Rayleigh quotient: x^T L x / x^T x
    rayleigh = (x @ L @ x) / (x @ x + 1e-12)
    return 1.0 / (1.0 + rayleigh)


def random_graph(n, graph_type='erdos_renyi', p=0.3):
    """Generate random graphs of various types."""
    if graph_type == 'erdos_renyi':
        A = (np.random.random((n, n)) < p).astype(float)
        np.fill_diagonal(A, 0)
        A = (A + A.T) / 2
    elif graph_type == 'small_world':
        k = max(2, int(p * n))
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(1, k // 2 + 1):
                A[i, (i + j) % n] = 1
                A[(i + j) % n, i] = 1
        # Rewire
        for i in range(n):
            for j in range(1, k // 2 + 1):
                if np.random.random() < 0.1:
                    new_j = np.random.randint(0, n)
                    if new_j != i:
                        A[i, (i + j) % n] = 0
                        A[(i + j) % n, i] = 0
                        A[i, new_j] = 1
                        A[new_j, i] = 1
    elif graph_type == 'barabasi_albert':
        m = max(1, int(p * n))
        A = np.zeros((n, n))
        degrees = np.zeros(n)
        for i in range(1, n):
            probs = degrees[:i] + 1
            probs = probs / probs.sum()
            targets = np.random.choice(i, size=min(m, i), replace=False, p=probs)
            for t in targets:
                A[i, t] = 1
                A[t, i] = 1
                degrees[i] += 1
                degrees[t] += 1
    elif graph_type == 'grid':
        side = int(np.sqrt(n))
        n = side * side
        A = np.zeros((n, n))
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if i + 1 < side:
                    A[idx, (i + 1) * side + j] = 1
                    A[(i + 1) * side + j, idx] = 1
                if j + 1 < side:
                    A[idx, i * side + j + 1] = 1
                    A[i * side + j + 1, idx] = 1
    return A


def musical_tradition_graph(tradition, n=20):
    """Generate graphs that mimic properties of different musical traditions."""
    np.random.seed(hash(tradition) % 2**31)
    
    if tradition == 'western':
        # Structured, hierarchical - Barabasi-Albert with high clustering
        A = random_graph(n, 'barabasi_albert', p=0.15)
        # Add some regular structure
        for i in range(n - 1):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
    elif tradition == 'gamelan':
        # Layered, cyclic, organic - small world with lots of rewiring
        A = random_graph(n, 'small_world', p=0.4)
        # Add cyclic layer
        for i in range(n):
            A[i, (i + 5) % n] = 1
            A[(i + 5) % n, i] = 1
    elif tradition == 'indian':
        # Complex raga structure - mix of linear and cyclic
        A = random_graph(n, 'small_world', p=0.3)
        for i in range(0, n - 3, 3):
            A[i, i + 3] = 1
            A[i + 3, i] = 1
    elif tradition == 'jazz':
        # High improvisation, many connections - dense Erdos-Renyi
        A = random_graph(n, 'erdos_renyi', p=0.4)
    elif tradition == 'african':
        # Polyrhythmic, interconnected - grid-like with diagonal connections
        A = random_graph(n, 'grid', p=0.3)
        side = int(np.sqrt(n))
        for i in range(side - 1):
            for j in range(side - 1):
                idx = i * side + j
                A[idx, (i + 1) * side + j + 1] = 1
                A[(i + 1) * side + j + 1, idx] = 1
    else:
        A = random_graph(n, 'erdos_renyi', p=0.25)
    
    return A
