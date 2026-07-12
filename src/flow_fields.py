"""Laplacian Flow Fields: Particle art from Laplacian eigenvectors."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import colorsys

from .graph_utils import (spectral_decomposition, conservation_ratio,
                           random_graph, musical_tradition_graph)
from .colors import harmonic_gradient


def generate_flow_field(A, title="Laplacian Flow", filename="flow.png",
                        n_particles=2000, steps=80, size=10, dpi=200,
                        bg_color='#050510'):
    """Generate particle-based flow art from Laplacian eigenvectors."""
    n = A.shape[0]
    vals, vecs = spectral_decomposition(A, k=min(n - 2, 8))
    
    # Conservation ratio
    x_attr = np.random.randn(n)
    cr = conservation_ratio(A, x_attr)
    
    # Graph node positions in 2D using first two non-trivial eigenvectors
    if vecs.shape[1] < 3:
        return
    node_x = vecs[:, 1]
    node_y = vecs[:, 2]
    
    # Normalize to [-1, 1]
    node_x = 2 * (node_x - node_x.min()) / (node_x.max() - node_x.min() + 1e-12) - 1
    node_y = 2 * (node_y - node_y.min()) / (node_y.max() - node_y.min() + 1e-12) - 1
    
    fig, ax = plt.subplots(figsize=(size, size), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Define flow field from Laplacian eigenvectors
    # Use higher eigenvectors for swirling flow

    def flow_field(px, py):
        """Compute flow direction at point (px, py) from graph nodes."""
        dx, dy = 0.0, 0.0
        for k in range(min(3, vecs.shape[1] - 1)):
            for i in range(n):
                dist_sq = (px - node_x[i])**2 + (py - node_y[i])**2 + 0.01
                weight = np.exp(-3 * dist_sq)
                # Flow from eigenvector k
                if k < vecs.shape[0]:
                    ev_k = vecs[i, min(k + 1, vecs.shape[1] - 1)]
                else:
                    ev_k = 0
                # Rotational component (perpendicular to radial)
                rx, ry = px - node_x[i], py - node_y[i]
                r = np.sqrt(rx**2 + ry**2) + 1e-6
                # Perpendicular direction
                perp_x, perp_y = -ry / r, rx / r
                dx += weight * ev_k * perp_x * (1 + k * 0.5)
                dy += weight * ev_k * perp_y * (1 + k * 0.5)
        
        # Normalize
        mag = np.sqrt(dx**2 + dy**2) + 1e-8
        return dx / mag, dy / mag
    
    # Generate particle traces (vectorized for speed)
    np.random.seed(42)
    base_hue, hue_spread = harmonic_gradient(cr, phase=0.3)
    
    # Precompute node influence weights
    n_ev = min(3, vecs.shape[1] - 1)
    ev_coeffs = np.zeros(n)
    for k in range(n_ev):
        col = min(k + 1, vecs.shape[1] - 1)
        ev_coeffs += vecs[:, col] * (1 + k * 0.5)
    
    # Batch process particles
    px = np.random.uniform(-1.3, 1.3, n_particles)
    py = np.random.uniform(-1.3, 1.3, n_particles)
    noise = 0.02 * (1.1 - cr)
    
    all_segments = []
    all_colors = []
    
    for p in range(n_particles):
        trail = [(px[p], py[p])]
        cpx, cpy = px[p], py[p]
        
        for s in range(steps):
            # Simplified flow: influence from all nodes
            dx_arr = cpx - node_x
            dy_arr = cpy - node_y
            dist_sq = dx_arr**2 + dy_arr**2 + 0.01
            weights = np.exp(-3 * dist_sq)
            r_arr = np.sqrt(dist_sq)
            
            # Perpendicular flow
            perp_x = -dy_arr / r_arr
            perp_y = dx_arr / r_arr
            
            fx = np.sum(weights * ev_coeffs * perp_x)
            fy = np.sum(weights * ev_coeffs * perp_y)
            mag = np.sqrt(fx**2 + fy**2) + 1e-8
            fx /= mag
            fy /= mag
            
            cpx += fx * 0.015 + np.random.randn() * noise
            cpy += fy * 0.015 + np.random.randn() * noise
            
            if abs(cpx) > 1.5 or abs(cpy) > 1.5:
                break
            trail.append((cpx, cpy))
        
        if len(trail) > 2:
            arr = np.array(trail)
            segs = np.column_stack([arr[:-1], arr[1:]]).reshape(-1, 2, 2)
            all_segments.append(segs)
            
            n_seg = len(segs)
            t = np.linspace(0, 1, n_seg)
            h = (base_hue + hue_spread * t * 0.5 + 0.1 * np.sin(p * 0.1)) % 1.0
            alpha = 0.02 + 0.6 * t * (1.0 - t)
            sat_arr = np.full(n_seg, 0.5 + 0.5 * cr)
            val_arr = 0.5 + 0.5 * t
            
            for si in range(n_seg):
                r, g, b = colorsys.hsv_to_rgb(h[si], sat_arr[si], val_arr[si])
                all_colors.append((r, g, b, float(alpha[si])))
    
    segments = []
    for seg_group in all_segments:
        segments.extend(seg_group)
    
    if segments:
        lc = LineCollection(segments, colors=all_colors, linewidths=0.5)
        ax.add_collection(lc)
    
    # Draw graph nodes as subtle glowing points
    for i in range(n):
        for k in range(min(3, vecs.shape[1] - 1)):
            ev = abs(vecs[i, min(k + 1, vecs.shape[1] - 1)])
            h = (base_hue + k * 0.15) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
            ax.plot(node_x[i], node_y[i], 'o', color=(r, g, b, 0.3),
                   markersize=3 + 5 * ev)
        ax.plot(node_x[i], node_y[i], 'o', color='white', markersize=1.5, zorder=10)
    
    ax.set_title(title, color='white', fontsize=14, pad=15,
                fontfamily='serif', fontstyle='italic')
    
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor=bg_color,
               edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {filename}")
    return filename


def generate_flow_gallery(output_dir="output"):
    """Generate flow fields from various graphs."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    configs = [
        ('small_world', 'Laplacian Vortex', 'flow_01_vortex.png', '#050510'),
        ('barabasi_albert', 'Scale-Free Currents', 'flow_02_currents.png', '#08050f'),
        ('erdos_renyi', 'Random Walk Dreams', 'flow_03_dreams.png', '#050808'),
    ]
    
    for gtype, title, fname, bg in configs:
        print(f"Generating: {title}")
        A = random_graph(20, gtype, p=0.3)
        generate_flow_field(A, title=title, filename=os.path.join(output_dir, fname),
                           bg_color=bg, n_particles=500, steps=50)
    
    for trad in ['western', 'gamelan']:
        print(f"Generating: {trad.title()} Flow")
        A = musical_tradition_graph(trad, n=20)
        generate_flow_field(A, title=f"{trad.title()} Tradition Flow",
                           filename=os.path.join(output_dir, f'flow_04_{trad}.png'),
                           n_particles=600, steps=50)
