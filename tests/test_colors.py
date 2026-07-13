"""Tests for src.colors color/palette logic."""
import numpy as np
import pytest

from src.colors import (conservation_colormap, spectral_palette,
                        harmonic_gradient, tradition_colors, hex_to_rgb)


def test_hex_to_rgb_known_values():
    assert hex_to_rgb('#000000') == (0.0, 0.0, 0.0)
    assert hex_to_rgb('#ffffff') == (1.0, 1.0, 1.0)
    assert hex_to_rgb('#ff0000') == (1.0, 0.0, 0.0)
    # works without leading '#'
    assert hex_to_rgb('00ff00') == (0.0, 1.0, 0.0)


@pytest.mark.parametrize('tradition',
                         ['western', 'gamelan', 'indian', 'jazz', 'african'])
def test_tradition_colors_returns_five(tradition):
    pal = tradition_colors(tradition)
    assert len(pal) == 5
    for c in pal:
        assert c.startswith('#') and len(c) == 7
    # all parse cleanly
    for c in pal:
        rgb = hex_to_rgb(c)
        assert len(rgb) == 3
        assert all(0.0 <= ch <= 1.0 for ch in rgb)


def test_tradition_colors_unknown_falls_back_to_western():
    assert tradition_colors('k-pop') == tradition_colors('western')


def test_conservation_colormap_shape_and_range():
    cols = conservation_colormap(0.5, n_colors=64)
    assert len(cols) == 64
    for r, g, b in cols:
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0


def test_conservation_colormap_count_arg():
    assert len(conservation_colormap(0.9, n_colors=10)) == 10
    assert len(conservation_colormap(0.1, n_colors=200)) == 200


def test_spectral_palette_length_matches_eigenvalues():
    vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    pal = spectral_palette(vals, conservation=0.7)
    assert len(pal) == len(vals)
    for r, g, b in pal:
        assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0


def test_spectral_palette_handles_zero_eigenvalues():
    # All-zero eigenvalues must not divide by zero.
    vals = np.zeros(5)
    pal = spectral_palette(vals, conservation=0.5)
    assert len(pal) == 5


@pytest.mark.parametrize('cr,expected_band', [
    (0.9, 'analogous'),
    (0.5, 'triadic'),
    (0.2, 'complementary'),
])
def test_harmonic_gradient_thresholds(cr, expected_band):
    base_hue, spread = harmonic_gradient(cr, phase=0.2)
    assert 0.0 <= base_hue < 1.0
    if expected_band == 'analogous':
        assert spread == pytest.approx(0.1)
    elif expected_band == 'triadic':
        assert spread == pytest.approx(0.33)
    else:
        assert spread == pytest.approx(0.5)


def test_harmonic_gradient_phase_wraps():
    base_hue, _ = harmonic_gradient(0.6, phase=1.2)
    assert 0.0 <= base_hue < 1.0
