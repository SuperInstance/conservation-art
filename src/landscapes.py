"""Eigenvalue Landscapes: 3D terrain from spectral properties."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import colorsys

from .graph_utils import (spectral_decomposition, conservation_ratio,
                           random_graph, musical_tradition_graph)


def generate_landscape(A, title="Eigenvalue Landscape", filename="landscape.png",
                       size=10, dpi=200):
    """Generate 3D terrain from graph eigenvectors."""
    n = A.shape[0]
    vals, vecs = spectral_decomposition(A, k=min(n - 2, 12))
    
    # Use eigenvectors 2 and 3 as X-Y coordinates, attribute as Z
    if vecs.shape[1] < 3:
        return
    
    x = vecs[:, 1]
    y = vecs[:, 2]
    
    fig = plt.figure(figsize=(size, size), facecolor='#0a0a1a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')
    
    # Create terrain surface from spectral data
    n_surf = 50
    u = np.linspace(x.min() - 0.1, x.max() + 0.1, n_surf)
    v = np.linspace(y.min() - 0.1, y.max() + 0.1, n_surf)
    U, V = np.meshgrid(u, v)
    
    # Interpolate Z from eigenvector combination
    from scipy.interpolate import RBFInterpolator
    try:
        coords = np.column_stack([x, y])
        z_vals = np.array([conservation_ratio(A, vecs[:, i]) for i in range(min(8, vecs.shape[1]))])
        z_nodes = np.mean(z_vals[:min(3, len(z_vals))])
        z_target = np.full(n, z_nodes)
        
        rbf = RBFInterpolator(coords, z_target, kernel='thin_plate_spline')
        grid_coords = np.column_stack([U.ravel(), V.ravel()])
        Z = rbf(grid_coords).reshape(U.shape)
    except Exception:
        Z = np.sin(3 * U) * np.cos(3 * V) * 0.5
    
    # Color by conservation gradient
    x_attr = np.random.randn(n)
    cr = conservation_ratio(A, x_attr)
    
    # Custom colormap based on conservation
    base_hue = 0.55 + 0.15 * cr
    colors_list = []
    for t in np.linspace(0, 1, 256):
        h = (base_hue + 0.3 * t) % 1.0
        s = 0.4 + 0.5 * t
        v = 0.2 + 0.7 * t
        colors_list.append(colorsys.hsv_to_rgb(h, s, v))
    
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors_list)
    
    ax.plot_surface(U, V, Z, cmap=cmap, alpha=0.7, 
                   rstride=1, cstride=1, linewidth=0,
                   antialiased=True)
    
    # Plot graph nodes as scatter
    node_z = np.array([conservation_ratio(A, vecs[:, i]) for i in range(min(3, vecs.shape[1]))])
    node_z_avg = np.full(n, np.mean(node_z))
    ax.scatter(x, y, node_z_avg, c='white', s=20, alpha=0.8, zorder=10,
               edgecolors='cyan', linewidth=0.5)
    
    # Draw graph edges
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                ax.plot([x[i], x[j]], [y[i], y[j]], 
                       [node_z_avg[i], node_z_avg[j]],
                       color='cyan', alpha=0.15, linewidth=0.5)
    
    ax.set_axis_off()
    ax.view_init(elev=35, azim=45)
    
    ax.set_title(title, color='white', fontsize=14, pad=20,
                fontfamily='serif', fontstyle='italic')
    
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='#0a0a1a',
               edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {filename}")
    return filename


def generate_landscape_gallery(output_dir="output"):
    """Generate landscapes from different graph structures."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    configs = [
        ('barabasi_albert', 'Scale-Free Topology', 'landscape_01_scalefree.png'),
        ('small_world', 'Small-World Terrain', 'landscape_02_smallworld.png'),
        ('grid', 'Lattice Prairie', 'landscape_03_lattice.png'),
    ]
    
    for gtype, title, fname in configs:
        print(f"Generating: {title}")
        A = random_graph(36, gtype, p=0.3)
        generate_landscape(A, title=title, filename=os.path.join(output_dir, fname))
    
    for trad in ['gamelan', 'jazz']:
        print(f"Generating: {trad.title()} Landscape")
        A = musical_tradition_graph(trad, n=36)
        generate_landscape(A, title=f"{trad.title()} Spectral Terrain",
                          filename=os.path.join(output_dir, f'landscape_04_{trad}.png'))
