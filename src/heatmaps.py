"""Conservation Heat Maps: Multi-scale conservation as color gradient."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .graph_utils import (spectral_decomposition, conservation_ratio, random_graph)


def generate_heatmap(A, title="Conservation Heat Map", filename="heatmap.png",
                     size=10, dpi=200):
    """Generate a heat map showing conservation ratio across scales."""
    n = A.shape[0]
    vals, vecs = spectral_decomposition(A, k=min(n - 2, 12))
    
    # Compute conservation at multiple scales
    n_scales = min(vecs.shape[1] - 1, 8)
    cr_values = []
    for k in range(1, n_scales + 1):
        x = vecs[:, k]
        cr = conservation_ratio(A, x)
        cr_values.append(cr)
    
    # Create 2D grid where we visualize conservation
    grid_size = 100
    grid = np.zeros((grid_size, grid_size))
    
    # Interpolate conservation values across grid using spectral structure
    for i in range(grid_size):
        for j in range(grid_size):
            x_norm = (i / grid_size - 0.5) * 2
            y_norm = (j / grid_size - 0.5) * 2
            r = np.sqrt(x_norm**2 + y_norm**2)
            
            # Each ring represents a different scale
            scale_idx = int(r * n_scales / 1.42)
            if scale_idx < n_scales:
                # Base value from conservation at this scale
                base_cr = cr_values[scale_idx]
                # Add angular variation from eigenvectors
                if vecs.shape[1] > scale_idx + 1:
                    angle = np.arctan2(y_norm, x_norm)
                    n_nodes = vecs.shape[0]
                    angular_idx = int((angle + np.pi) / (2 * np.pi) * (n_nodes - 1))
                    angular_idx = min(angular_idx, n_nodes - 1)
                    variation = abs(vecs[angular_idx, min(scale_idx + 1, vecs.shape[1] - 1)])
                    grid[i, j] = base_cr + 0.3 * variation
                else:
                    grid[i, j] = base_cr
            else:
                grid[i, j] = 0.1
    
    # Custom colormap: deep purple → teal → gold → white
    colors_list = [
        (0.05, 0.02, 0.15),  # Deep purple
        (0.1, 0.1, 0.3),     # Dark blue
        (0.0, 0.4, 0.5),     # Teal
        (0.2, 0.8, 0.4),     # Green
        (0.9, 0.8, 0.2),     # Gold
        (1.0, 0.4, 0.1),     # Orange (anomalous)
        (1.0, 0.95, 0.95),   # Near white
    ]
    cmap = LinearSegmentedColormap.from_list('conservation', colors_list, N=256)
    
    fig, ax = plt.subplots(figsize=(size, size), facecolor='#0a0a12')
    ax.set_facecolor('#0a0a12')
    
    ax.imshow(grid, cmap=cmap, interpolation='bilinear',
              extent=[-1, 1, -1, 1], vmin=0, vmax=1.5)
    
    # Overlay graph structure
    if vecs.shape[1] >= 3:
        node_x = vecs[:, 1]
        node_y = vecs[:, 2]
        node_x = (node_x - node_x.min()) / (node_x.max() - node_x.min() + 1e-12) * 1.6 - 0.8
        node_y = (node_y - node_y.min()) / (node_y.max() - node_y.min() + 1e-12) * 1.6 - 0.8
        
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] > 0:
                    ax.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]],
                           color='white', alpha=0.1, linewidth=0.3)
        
        ax.scatter(node_x, node_y, c='white', s=8, alpha=0.6, zorder=10, edgecolors='none')
    
    ax.axis('off')
    
    # Conservation scale annotation
    text_color = '#cccccc'
    ax.text(0.02, 0.98, f'Conservation: {np.mean(cr_values):.2f}',
           transform=ax.transAxes, color=text_color, fontsize=10,
           verticalalignment='top', fontfamily='monospace')
    ax.text(0.02, 0.94, f'Scales: {n_scales}',
           transform=ax.transAxes, color=text_color, fontsize=9,
           verticalalignment='top', fontfamily='monospace')
    
    ax.set_title(title, color='white', fontsize=14, pad=15,
                fontfamily='serif', fontstyle='italic')
    
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='#0a0a12',
               edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {filename}")
    return filename


def generate_heatmap_gallery(output_dir="output"):
    """Generate conservation heat maps from various graphs."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    configs = [
        ('barabasi_albert', 'Scale-Free Conservation', 'heatmap_01_scalefree.png'),
        ('small_world', 'Small-World Conservation', 'heatmap_02_smallworld.png'),
        ('erdos_renyi', 'Random Graph Conservation', 'heatmap_03_random.png'),
        ('grid', 'Lattice Conservation', 'heatmap_04_lattice.png'),
    ]
    
    for gtype, title, fname in configs:
        print(f"Generating: {title}")
        A = random_graph(25, gtype, p=0.3)
        generate_heatmap(A, title=title, filename=os.path.join(output_dir, fname))
