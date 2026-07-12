"""Tests for spectral graph utilities in src.graph_utils."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.graph_utils import (laplacian, conservation_ratio, spectral_decomposition,
                             random_graph, musical_tradition_graph, build_graph)


# --------------------------- musical_tradition_graph ---------------------------

TRADITIONS = ['western', 'gamelan', 'indian', 'jazz', 'african']


def _run_in_subprocess(snippet):
    """Run a code snippet in a fresh Python process and capture its stdout."""
    import subprocess
    cmd = [sys.executable, '-c',
           'import sys; sys.path.insert(0, %r)\n' % ROOT + snippet]
    out = subprocess.check_output(cmd, text=True)
    return out.strip()


def test_tradition_graph_reproducible_across_processes():
    """Same tradition name -> identical graph in independent Python processes.

    Regression for the hash()-based seed, which was randomized per process by
    CPython's PYTHONHASHSEED and produced different art on every run.
    """
    snippet = (
        'import numpy as np\n'
        'from src.graph_utils import musical_tradition_graph\n'
        'g = musical_tradition_graph("western", n=25)\n'
        'print(g.tobytes().hex())\n'
    )
    a = _run_in_subprocess(snippet)
    b = _run_in_subprocess(snippet)
    c = _run_in_subprocess(snippet)
    assert a == b == c, "tradition graph must be deterministic across processes"


@pytest.mark.parametrize('tradition', TRADITIONS)
def test_tradition_graph_valid_for_all_traditions(tradition):
    A = musical_tradition_graph(tradition, n=20)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    # symmetric and zero diagonal for all types
    assert np.allclose(A, A.T), f"{tradition} graph must be symmetric"
    assert np.all(np.diag(A) == 0), f"{tradition} graph must have zero diagonal"
    assert A.sum() > 0, f"{tradition} graph must have at least one edge"


def test_unknown_tradition_falls_back():
    A = musical_tradition_graph('does-not-exist', n=12)
    assert A.shape == (12, 12)
    assert np.allclose(A, A.T)


# --------------------------- random_graph ---------------------------

@pytest.mark.parametrize('gtype', ['erdos_renyi', 'small_world',
                                   'barabasi_albert', 'grid'])
def test_random_graph_basic_invariants(gtype):
    n = 25
    A = random_graph(n, gtype, p=0.3)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    # symmetric and zero diagonal for all types
    assert np.allclose(A, A.T)
    assert np.all(np.diag(A) == 0)


def test_random_graph_honors_requested_size():
    # Regression: the grid path used to silently truncate to floor(sqrt(n))^2,
    # returning fewer nodes than requested (e.g. 16 for n=20). Every graph type
    # must now return exactly n nodes.
    for n in [7, 20, 30, 36, 50]:
        for gtype in ['erdos_renyi', 'small_world', 'barabasi_albert', 'grid']:
            A = random_graph(n, gtype, p=0.3)
            assert A.shape == (n, n), f"{gtype}(n={n}) -> {A.shape}"


def test_grid_graph_has_internal_lattice_edges():
    # A grid must actually contain grid-style 4-neighbor edges.
    A = random_graph(25, 'grid', p=0.3)
    # Interior corner node 6 (row 1, col 1 of a 5x5) should connect to 4 neighbors.
    assert A[6].sum() == 4


def test_barabasi_albert_is_connected():
    # BA model is connected by construction when starting from one node.
    np.random.seed(0)
    A = random_graph(20, 'barabasi_albert', p=0.2)
    # BFS reachability from node 0
    seen = set()
    frontier = [0]
    while frontier:
        nxt = frontier.pop()
        if nxt in seen:
            continue
        seen.add(nxt)
        frontier.extend(np.flatnonzero(A[nxt]).tolist())
    assert len(seen) == A.shape[0]
