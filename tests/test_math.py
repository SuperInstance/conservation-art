"""Numerical correctness tests for the spectral math in src.graph_utils.

These spot-check claims against known-closed-form results.
"""
import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.graph_utils import (laplacian, conservation_ratio, spectral_decomposition)


# --------------------------- laplacian ---------------------------

def test_laplacian_path_graph_known_values():
    # Path graph P3: degrees [1, 2, 1]; normalized L off-diagonal = -1/sqrt(d_i d_j)
    A = np.array([[0., 1., 0.],
                  [1., 0., 1.],
                  [0., 1., 0.]])
    L = laplacian(A)
    assert np.allclose(np.diag(L), 1.0)
    assert np.isclose(L[0, 1], -1.0 / np.sqrt(2))
    assert np.isclose(L[1, 2], -1.0 / np.sqrt(2))
    assert np.isclose(L[0, 2], 0.0)


def test_laplacian_is_psd():
    # x^T L x >= 0 for all x (normalized Laplacian is positive semidefinite)
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = rng.integers(3, 12)
        A = (rng.random((n, n)) < 0.4).astype(float)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        L = laplacian(A)
        x = rng.standard_normal(n)
        assert x @ L @ x >= -1e-12


def test_laplacian_symmetric():
    rng = np.random.default_rng(1)
    A = (rng.random((6, 6)) < 0.5).astype(float)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    assert np.allclose(laplacian(A), laplacian(A).T)


def test_laplacian_isolated_node_no_warning():
    # Regression: isolated nodes used to emit a RuntimeWarning (divide by zero
    # from np.where evaluating 1/sqrt(0) before selecting the fallback).
    A = np.zeros((5, 5))
    A[0:3, 0:3] = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    # nodes 3 and 4 are isolated
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        L = laplacian(A)
    # Isolated node -> identity row: diag 1, off-diag 0
    assert np.isclose(L[3, 3], 1.0)
    assert np.all(L[3, :] == np.array([0, 0, 0, 1, 0]))
    assert np.isclose(L[4, 4], 1.0)


# --------------------------- conservation_ratio ---------------------------

def test_conservation_ratio_known_2node_graph():
    # A = [[0,1],[1,0]] -> L = [[1,-1],[-1,1]]
    A = np.array([[0., 1.], [1., 0.]])
    # constant on a regular graph -> Rayleigh = 0 -> sigma = 1
    assert np.isclose(conservation_ratio(A, np.array([1., 1.])), 1.0)
    # alternating -> x^T L x = 4, x^T x = 2 -> Rayleigh = 2 -> sigma = 1/3
    assert np.isclose(conservation_ratio(A, np.array([1., -1.])), 1.0 / 3.0)


def test_conservation_ratio_range_and_ordering():
    rng = np.random.default_rng(2)
    A = (rng.random((10, 10)) < 0.3).astype(float)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    # Smooth signal (constant-ish) should score higher than a noisy one.
    smooth = np.ones(10) + 0.01 * rng.standard_normal(10)
    rough = rng.standard_normal(10)
    cr_smooth = conservation_ratio(A, smooth)
    cr_rough = conservation_ratio(A, rough)
    assert 0.0 < cr_smooth <= 1.0
    assert 0.0 < cr_rough <= 1.0
    assert cr_smooth > cr_rough


def test_conservation_ratio_respects_lower_bound():
    # For the normalized Laplacian, eigenvalues are in [0, 2], so the
    # Rayleigh quotient is in [0, 2] and sigma = 1/(1+rq) is in [1/3, 1].
    rng = np.random.default_rng(3)
    lowest = 1.0
    for _ in range(50):
        n = rng.integers(4, 15)
        A = (rng.random((n, n)) < 0.3).astype(float)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        x = rng.standard_normal(n)
        cr = conservation_ratio(A, x)
        assert cr >= 1.0 / 3.0 - 1e-9
        lowest = min(lowest, cr)
    assert lowest <= 1.0


# --------------------------- spectral_decomposition ---------------------------

def test_spectral_decomposition_eigenvalues_ascending_and_valid():
    rng = np.random.default_rng(4)
    A = (rng.random((12, 12)) < 0.3).astype(float)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    vals, vecs = spectral_decomposition(A, k=6)
    assert vals.shape == (6,)
    assert vecs.shape == (12, 6)
    assert np.all(np.diff(vals) >= -1e-9), "eigenvalues must be ascending"
    # Normalized Laplacian spectrum lies in [0, 2].
    assert vals.min() >= -1e-9
    assert vals.max() <= 2.0 + 1e-9


def test_spectral_decomposition_smallest_eigenvalue_near_zero():
    # Any connected graph has a single zero eigenvalue for the normalized
    # Laplacian (the constant vector on a regular-ish connected graph).
    # Use a complete graph (regular) so the constant vector is exact.
    n = 6
    A = np.ones((n, n)) - np.eye(n)
    vals, _ = spectral_decomposition(A, k=3)
    assert np.isclose(vals[0], 0.0, atol=1e-8)


def test_spectral_decomposition_tiny_graph_fallback():
    # n=2 -> k = min(n, n-2) = 0 -> falls back to full eigh.
    A = np.array([[0., 1.], [1., 0.]])
    vals, vecs = spectral_decomposition(A)
    assert vals.shape[0] == 2
    assert vecs.shape == (2, 2)
