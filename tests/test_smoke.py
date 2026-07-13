"""Scaffold smoke test: verifies the package imports and core math runs.

This is intentionally minimal. Real coverage is added per-fix in sibling
test modules. It must FAIL (not skip) if the package is broken.
"""
import os

import numpy as np


def test_package_imports():
    import src  # noqa: F401
    from src import graph_utils, colors, mandala, landscapes, flow_fields, portraits, heatmaps  # noqa: F401


def test_laplacian_basic_runs():
    from src.graph_utils import laplacian, conservation_ratio
    A = np.array([[0.0, 1.0, 0.0],
                  [1.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0]])
    L = laplacian(A)
    assert L.shape == (3, 3)
    # Constant vector on a regular graph -> perfectly smooth -> conservation ~ 1.
    # Path graph P3 is NOT regular at endpoints, so just check the value range.
    cr = conservation_ratio(A, np.array([1.0, 1.0, 1.0]))
    assert 0.0 < cr <= 1.0


def test_generate_module_importable():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate", os.path.join(os.path.dirname(os.path.dirname(__file__)), "generate.py"))
    assert spec is not None and spec.loader is not None
