"""Tradition Portraits: Abstract art for each musical tradition from Laplacian eigenvalues."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

from .graph_utils import (spectral_decomposition, musical_tradition_graph)
from .colors import tradition_colors, hex_to_rgb


def generate_tradition_portrait(tradition, n=30, filename="portrait.png",
                                size=10, dpi=200):
    """Generate abstract art portrait of a musical tradition."""
    A = musical_tradition_graph(tradition, n)
    vals, vecs = spectral_decomposition(A, k=min(n - 2, 10))
    palette = [hex_to_rgb(c) for c in tradition_colors(tradition)]
    
    fig, ax = plt.subplots(figsize=(size, size), facecolor='#0a0a12')
    ax.set_facecolor('#0a0a12')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    style_map = {
        'western': 'geometric',
        'gamelan': 'organic',
        'indian': 'spiral',
        'jazz': 'chaotic',
        'african': 'tribal',
    }
    style = style_map.get(tradition, 'geometric')
    
    if style == 'geometric':
        _draw_geometric(ax, A, vals, vecs, palette, n)
    elif style == 'organic':
        _draw_organic(ax, A, vals, vecs, palette, n)
    elif style == 'spiral':
        _draw_spiral(ax, A, vals, vecs, palette, n)
    elif style == 'chaotic':
        _draw_chaotic(ax, A, vals, vecs, palette, n)
    elif style == 'tribal':
        _draw_tribal(ax, A, vals, vecs, palette, n)
    
    title_map = {
        'western': 'Western Tradition\nStructured Harmonic Geometry',
        'gamelan': 'Gamelan Tradition\nLayered Organic Resonance',
        'indian': 'Indian Tradition\nRaga Spiral Complexity',
        'jazz': 'Jazz Tradition\nImprovisational Chaos',
        'african': 'African Tradition\nPolyrhythmic Interconnection',
    }
    
    ax.set_title(title_map.get(tradition, tradition.title()),
                color='white', fontsize=13, pad=15,
                fontfamily='serif', fontstyle='italic', linespacing=1.5)
    
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='#0a0a12',
               edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {filename}")
    return filename


def _node_positions(vals, vecs, n):
    """Get 2D node positions from spectral embedding."""
    if vecs.shape[1] < 3:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.cos(angles), np.sin(angles)
    x = vecs[:, 1]
    y = vecs[:, 2]
    x = 1.2 * (x - x.min()) / (x.max() - x.min() + 1e-12) - 0.6
    y = 1.2 * (y - y.min()) / (y.max() - y.min() + 1e-12) - 0.6
    return x, y


def _draw_geometric(ax, A, vals, vecs, palette, n):
    """Western: structured geometric patterns."""
    nx, ny = _node_positions(vals, vecs, n)
    
    # Concentric polygons
    for layer in range(min(6, len(vals))):
        ev = vals[layer]
        n_sides = max(3, int(4 + layer * 2))
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        r = 0.3 + 0.8 * layer / 6
        rotation = ev * np.pi
        
        x_pts = r * np.cos(angles + rotation)
        y_pts = r * np.sin(angles + rotation)
        x_pts = np.append(x_pts, x_pts[0])
        y_pts = np.append(y_pts, y_pts[0])
        
        color = palette[layer % len(palette)]
        alpha = 0.15 + 0.2 * (1 - layer / 6)
        ax.fill(x_pts, y_pts, color=(*color, alpha * 0.4), linewidth=0)
        ax.plot(x_pts, y_pts, color=(*color, 0.5 + 0.3 * alpha), linewidth=1.5)
    
    # Node connections as straight geometric lines
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                ax.plot([nx[i], nx[j]], [ny[i], ny[j]],
                       color=(*palette[0], 0.08), linewidth=0.3)
    
    # Nodes as small squares
    for i in range(n):
        c = palette[i % len(palette)]
        ax.plot(nx[i], ny[i], 's', color=(*c, 0.6), markersize=3)


def _draw_organic(ax, A, vals, vecs, palette, n):
    """Gamelan: layered organic curves."""
    theta = np.linspace(0, 2 * np.pi, 500)
    
    for layer in range(min(5, len(vals))):
        ev = vals[layer]
        freq = 3 + layer * 2
        r_base = 0.2 + 0.2 * layer
        
        # Organic wavy shape from eigenvalue
        r = r_base + 0.1 * np.sin(freq * theta + ev * 10) + 0.05 * np.cos((freq + 3) * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        color = palette[layer % len(palette)]
        ax.fill(x, y, color=(*color, 0.1), linewidth=0)
        ax.plot(x, y, color=(*color, 0.4 + 0.2 * layer / 5), linewidth=2 - 0.2 * layer)
    
    # Scattered organic nodes
    nx, ny = _node_positions(vals, vecs, n)
    for i in range(n):
        c = palette[i % len(palette)]
        circle = plt.Circle((nx[i], ny[i]), 0.03 + 0.02 * np.random.random(),
                           color=(*c, 0.4), linewidth=0)
        ax.add_patch(circle)
    
    # Curved edges
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                mid_x = (nx[i] + nx[j]) / 2 + np.random.randn() * 0.05
                mid_y = (ny[i] + ny[j]) / 2 + np.random.randn() * 0.05
                t = np.linspace(0, 1, 20)
                bx = (1 - t)**2 * nx[i] + 2 * (1 - t) * t * mid_x + t**2 * nx[j]
                by = (1 - t)**2 * ny[i] + 2 * (1 - t) * t * mid_y + t**2 * ny[j]
                ax.plot(bx, by, color=(*palette[1], 0.06), linewidth=0.5)


def _draw_spiral(ax, A, vals, vecs, palette, n):
    """Indian: spiral raga patterns."""
    theta = np.linspace(0, 8 * np.pi, 1000)
    
    for layer in range(min(4, len(vals))):
        ev = vals[layer]
        r = 0.02 * theta + 0.05 * np.sin(ev * theta * 3) * theta * 0.01
        
        x = r * np.cos(theta + layer * 0.5)
        y = r * np.sin(theta + layer * 0.5)
        
        color = palette[layer % len(palette)]
        
        # Draw as gradient line segments
        points = np.column_stack([x, y])
        segs = np.column_stack([points[:-1], points[1:]]).reshape(-1, 2, 2)
        colors = []
        for i in range(len(segs)):
            t = i / len(segs)
            alpha = 0.05 + 0.3 * np.sin(t * np.pi) ** 2
            colors.append((*color, alpha))
        
        lc = LineCollection(segs, colors=colors, linewidths=1)
        ax.add_collection(lc)
    
    # Decorative dots along spiral
    for i in range(0, len(theta), 50):
        r = 0.02 * theta[i]
        x = r * np.cos(theta[i])
        y = r * np.sin(theta[i])
        c = palette[(i // 50) % len(palette)]
        ax.plot(x, y, 'o', color=(*c, 0.3), markersize=2)


def _draw_chaotic(ax, A, vals, vecs, palette, n):
    """Jazz: chaotic, dense, energetic."""
    np.random.seed(42)
    nx = np.random.randn(n) * 0.5
    ny = np.random.randn(n) * 0.5
    
    # Dense edge network
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                # Jagged paths
                n_pts = 10
                t = np.linspace(0, 1, n_pts)
                xs = nx[i] * (1 - t) + nx[j] * t + np.random.randn(n_pts) * 0.05
                ys = ny[i] * (1 - t) + ny[j] * t + np.random.randn(n_pts) * 0.05
                color = palette[np.random.randint(len(palette))]
                ax.plot(xs, ys, color=(*color, 0.08), linewidth=0.5)
    
    # Energetic node splashes
    for i in range(n):
        for k in range(3):
            r = 0.05 + 0.03 * np.random.random()
            angle = np.random.random() * 2 * np.pi
            cx = nx[i] + r * np.cos(angle)
            cy = ny[i] + r * np.sin(angle)
            color = palette[k % len(palette)]
            circle = plt.Circle((cx, cy), 0.02, color=(*color, 0.2), linewidth=0)
            ax.add_patch(circle)
        ax.plot(nx[i], ny[i], 'o', color=(*palette[3 % len(palette)], 0.7), markersize=4)
    
    # Golden accent lines
    for _ in range(20):
        x1, y1 = np.random.randn(2) * 0.6
        x2, y2 = np.random.randn(2) * 0.6
        ax.plot([x1, x2], [y1, y2], color=(*palette[1], 0.1), linewidth=2)


def _draw_tribal(ax, A, vals, vecs, palette, n):
    """African: bold, rhythmic, interconnected."""
    actual_n = A.shape[0]
    nx, ny = _node_positions(vals, vecs, actual_n)
    
    # Bold circular patterns
    for layer in range(min(4, len(vals))):
        ev = vals[layer]
        r = 0.3 + 0.25 * layer
        # Tribal zigzag on circle
        n_teeth = int(8 + 4 * ev)
        angles = np.linspace(0, 2 * np.pi, n_teeth * 2 + 1)
        radii = r + 0.05 * np.array([(-1)**i for i in range(len(angles))])
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        color = palette[layer % len(palette)]
        ax.plot(x, y, color=(*color, 0.5), linewidth=2)
    
    # Bold edges
    for i in range(actual_n):
        for j in range(i + 1, actual_n):
            if A[i, j] > 0:
                color = palette[(i + j) % len(palette)]
                ax.plot([nx[i], nx[j]], [ny[i], ny[j]],
                       color=(*color, 0.15), linewidth=1.5)
    
    # Bold triangular nodes
    for i in range(actual_n):
        size = 0.04 + 0.02 * np.random.random()
        c = palette[i % len(palette)]
        triangle = Polygon([
            [nx[i], ny[i] + size],
            [nx[i] - size * 0.866, ny[i] - size * 0.5],
            [nx[i] + size * 0.866, ny[i] - size * 0.5],
        ], closed=True, color=(*c, 0.6), linewidth=0)
        ax.add_patch(triangle)
        ax.plot(nx[i], ny[i], 'o', color='white', markersize=1.5, zorder=10)


def generate_portrait_gallery(output_dir="output"):
    """Generate tradition portraits."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for trad in ['western', 'gamelan', 'indian', 'jazz', 'african']:
        print(f"Generating: {trad.title()} Portrait")
        generate_tradition_portrait(trad, n=30,
                                    filename=os.path.join(output_dir, f'portrait_{trad}.png'))
