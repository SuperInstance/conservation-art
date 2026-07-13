"""End-to-end smoke tests for every art generator.

These are deliberately NOT unit tests of matplotlib internals. Each test
drives a real generator end-to-end and asserts a valid, non-blank PNG was
written to disk. A blank or missing file -> the test fails. This is the
guard against 'fake-green' CI where tests pass without exercising output.
"""
import os

import numpy as np
import pytest
from PIL import Image

from src.graph_utils import random_graph


def _assert_real_png(path, min_bytes=2048, min_distinct_colors=8):
    """Open a PNG and verify it is a real, non-blank image."""
    assert os.path.exists(path), f"generator produced no file: {path}"
    assert os.path.getsize(path) >= min_bytes, \
        f"{path} is suspiciously small ({os.path.getsize(path)} bytes)"
    with Image.open(path) as im:
        assert im.format == 'PNG', f"{path} is not a PNG (got {im.format})"
        w, h = im.size
        assert w > 100 and h > 100, f"{path} too small: {im.size}"
        rgb = im.convert('RGB')
        colors = rgb.getcolors(maxcolors=1_000_000)
        assert colors is not None, f"{path} has >1e6 colors (unexpected)"
        assert len(colors) >= min_distinct_colors, \
            f"{path} is near-blank: only {len(colors)} distinct colors"


def _small_graph(n=14, seed=0):
    np.random.seed(seed)
    A = random_graph(n, 'small_world', p=0.3)
    return A


# --------------------------- mandala ---------------------------

def test_mandala_renders_real_png(tmp_path):
    from src.mandala import generate_mandala
    out = tmp_path / 'm.png'
    generate_mandala(_small_graph(), title='t',
                     filename=str(out), size=4, dpi=72)
    _assert_real_png(out)


# --------------------------- landscape ---------------------------

def test_landscape_renders_real_png(tmp_path):
    from src.landscapes import generate_landscape
    out = tmp_path / 'l.png'
    generate_landscape(_small_graph(16), title='t',
                       filename=str(out), size=4, dpi=72)
    _assert_real_png(out)


# --------------------------- flow field ---------------------------

def test_flow_field_renders_real_png(tmp_path):
    from src.flow_fields import generate_flow_field
    out = tmp_path / 'f.png'
    generate_flow_field(_small_graph(12), title='t',
                        filename=str(out), size=4, dpi=72,
                        n_particles=80, steps=20)
    _assert_real_png(out, min_distinct_colors=2)


# --------------------------- portraits ---------------------------

@pytest.mark.parametrize('tradition',
                         ['western', 'gamelan', 'indian', 'jazz', 'african'])
def test_portrait_renders_real_png(tradition, tmp_path):
    from src.portraits import generate_tradition_portrait
    out = tmp_path / f'p_{tradition}.png'
    generate_tradition_portrait(tradition, n=16,
                                filename=str(out), size=4, dpi=72)
    _assert_real_png(out, min_distinct_colors=2)


# --------------------------- heatmap ---------------------------

def test_heatmap_renders_real_png(tmp_path):
    from src.heatmaps import generate_heatmap
    out = tmp_path / 'h.png'
    generate_heatmap(_small_graph(16), title='t',
                     filename=str(out), size=4, dpi=72)
    _assert_real_png(out)


# --------------------------- error-path: tiny / degenerate graphs ---------------------------

def test_generators_handle_disconnected_graph(tmp_path):
    """A graph with two disconnected components must still render."""
    from src.mandala import generate_mandala
    A = np.zeros((12, 12))
    A[0:6, 0:6] = (np.random.random((6, 6)) < 0.5)
    A[6:12, 6:12] = (np.random.random((6, 6)) < 0.5)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    out = tmp_path / 'disc.png'
    generate_mandala(A, title='t', filename=str(out), size=4, dpi=72)
    _assert_real_png(out)


def test_generators_handle_graph_with_isolated_nodes(tmp_path):
    """Graphs containing isolated (degree-0) nodes must render without warnings."""
    import warnings
    from src.mandala import generate_mandala
    A = np.zeros((12, 12))
    A[0:8, 0:8] = (np.random.random((8, 8)) < 0.4)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    # nodes 8..11 are isolated
    out = tmp_path / 'iso.png'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        generate_mandala(A, title='t', filename=str(out), size=4, dpi=72)
    _assert_real_png(out, min_distinct_colors=2)
