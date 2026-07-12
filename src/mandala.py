"""Spectral Mandala: Eigenvectors as angles/radii, eigenvalues as layers."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import colorsys

from .graph_utils import (spectral_decomposition, conservation_ratio,
                           random_graph, musical_tradition_graph)
from .colors import spectral_palette


def generate_mandala(A, title="Spectral Mandala", filename="mandala.png",
                     size=10, dpi=200, bg_color='#0a0a1a'):
    """Generate a mandala from graph spectral decomposition."""
    n = A.shape[0]
    vals, vecs = spectral_decomposition(A, k=min(n - 2, 16))
    
    # Generate attributes for conservation ratio
    x = np.random.randn(n)
    cr = conservation_ratio(A, x)
    
    fig, ax = plt.subplots(1, 1, figsize=(size, size), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    n_layers = min(len(vals), 10)
    palette = spectral_palette(vals, cr)
    
    for layer in range(n_layers - 1, 0, -1):
        eigenvalue = vals[layer]
        vec = vecs[:, layer]
        
        # Normalize eigenvector for mandala geometry
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = np.abs(vec) / (np.max(np.abs(vec)) + 1e-12)
        
        # Layer radius scales with eigenvalue
        layer_scale = 0.3 + 0.8 * (1.0 - eigenvalue / (vals[-1] + 1e-6))
        
        # Phase from eigenvector sign pattern
        phase = np.cumsum(vec) / (np.sum(np.abs(vec)) + 1e-12)
        angles_offset = angles + phase[0] * np.pi
        
        x_pts = layer_scale * radii * np.cos(angles_offset + layer * 0.2)
        y_pts = layer_scale * radii * np.sin(angles_offset + layer * 0.2)
        
        # Close the shape
        x_pts = np.append(x_pts, x_pts[0])
        y_pts = np.append(y_pts, y_pts[0])
        
        # Color from palette with alpha
        r, g, b = palette[layer % len(palette)]
        alpha = 0.15 + 0.4 * (1.0 - layer / n_layers)
        
        ax.fill(x_pts, y_pts, color=(r, g, b, alpha * 0.5), linewidth=0)
        ax.plot(x_pts, y_pts, color=(r, g, b, 0.6 + 0.4 * alpha), 
                linewidth=1.5 - layer * 0.1, zorder=layer + 2)
        
        # Inner connections between nodes
        for i in range(n):
            for j in range(i + 1, min(i + 3, n)):
                if A[i, j] > 0:
                    alpha_line = 0.08 + 0.15 * radii[i] * radii[j]
                    ax.plot([x_pts[i], x_pts[j]], [y_pts[i], y_pts[j]],
                           color=(r, g, b, alpha_line), linewidth=0.5, zorder=1)
    
    # Central node
    center_color = colorsys.hsv_to_rgb(0.6 * cr, 0.8, 0.9)
    ax.plot(0, 0, 'o', color=center_color, markersize=6, zorder=100)
    
    # Outer glow ring
    theta = np.linspace(0, 2 * np.pi, 200)
    for r_val in [1.2, 1.3, 1.4]:
        alpha = 0.1 * (1.5 - r_val)
        ax.plot(r_val * np.cos(theta), r_val * np.sin(theta),
               color=(*center_color, alpha), linewidth=0.5)
    
    ax.set_title(title, color='white', fontsize=14, pad=15,
                fontfamily='serif', fontstyle='italic')
    
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor=bg_color,
               edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {filename}")
    return filename


def generate_mandala_gallery(output_dir="output"):
    """Generate multiple mandalas from different graph types."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    configs = [
        ('erdos_renyi', 'Erdős-Rényi Constellation', 'mandala_01_erdos.png'),
        ('barabasi_albert', 'Barabási-Albert Nebula', 'mandala_02_barabasi.png'),
        ('small_world', 'Small World Chakra', 'mandala_03_smallworld.png'),
        ('grid', 'Grid Lattice Bloom', 'mandala_04_grid.png'),
    ]
    
    for gtype, title, fname in configs:
        print(f"Generating: {title}")
        A = random_graph(30, gtype, p=0.3)
        generate_mandala(A, title=title, filename=os.path.join(output_dir, fname))
    
    # Tradition-specific mandalas
    for trad in ['western', 'gamelan', 'indian', 'jazz', 'african']:
        print(f"Generating: {trad.title()} Mandala")
        A = musical_tradition_graph(trad, n=25)
        generate_mandala(A, 
                        title=f"{trad.title()} Tradition Mandala",
                        filename=os.path.join(output_dir, f'mandala_05_{trad}.png'),
                        bg_color='#0d0d1a' if trad == 'jazz' else '#0a0a1a')
