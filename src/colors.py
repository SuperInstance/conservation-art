"""Color palettes derived from conservation ratios and spectral properties."""
import numpy as np
import colorsys


def conservation_colormap(ratio, n_colors=256):
    """Generate a colormap where conservation ratio drives the hue.
    
    High conservation (smooth) → cool blues/purples
    Low conservation (anomalous) → warm reds/oranges
    """
    colors = []
    for i in range(n_colors):
        t = i / (n_colors - 1)
        # Map conservation ratio to hue
        hue = (1.0 - ratio) * 0.0 + ratio * 0.7  # red to blue
        hue = (hue + t * 0.1) % 1.0
        sat = 0.6 + 0.4 * abs(np.sin(t * np.pi))
        val = 0.3 + 0.7 * t
        colors.append(colorsys.hsv_to_rgb(hue, sat, val))
    return colors


def spectral_palette(eigenvalues, conservation):
    """Generate a color palette from spectral properties.
    
    Each eigenvalue defines a base hue, conservation controls saturation.
    """
    n = len(eigenvalues)
    palette = []
    for i, ev in enumerate(eigenvalues):
        # Eigenvalue maps to hue (normalized)
        hue = (ev / (max(eigenvalues) + 1e-6)) * 0.8
        hue = hue % 1.0
        # Conservation controls saturation and value
        sat = 0.5 + 0.5 * conservation
        val = 0.4 + 0.6 * (1.0 - ev / (max(eigenvalues) + 1e-6))
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((r, g, b))
    return palette


def harmonic_gradient(conservation, phase=0.0):
    """Generate harmonious gradient based on conservation ratio.
    
    Conservation near 1.0 → analogous colors (harmonious)
    Conservation near 0.0 → complementary colors (tension)
    """
    base_hue = (phase % 1.0)
    if conservation > 0.7:
        # Analogous: tight hue range
        hue_spread = 0.1
    elif conservation > 0.4:
        # Triadic: moderate spread
        hue_spread = 0.33
    else:
        # Complementary: wide spread
        hue_spread = 0.5
    
    return base_hue, hue_spread


def tradition_colors(tradition):
    """Color schemes inspired by musical traditions."""
    palettes = {
        'western': ['#2C3E50', '#E74C3C', '#ECF0F1', '#3498DB', '#F39C12'],
        'gamelan': ['#1A535C', '#4ECDC4', '#F7FFF7', '#FF6B6B', '#FFE66D'],
        'indian': ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1'],
        'jazz': ['#2D2D2D', '#D4AF37', '#C41E3A', '#003153', '#F5F5DC'],
        'african': ['#E2725B', '#FFD700', '#228B22', '#8B0000', '#FF8C00'],
    }
    return palettes.get(tradition, palettes['western'])


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
