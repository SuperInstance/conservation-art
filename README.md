# Conservation-Aware Generative Art

Beautiful visual art generated from the spectral properties of graphs — eigenvectors, eigenvalues, Laplacians, and conservation ratios.

## The Concept

Every graph has a spectral fingerprint: the eigenvalues and eigenvectors of its Laplacian matrix encode deep structural information. This project transforms those mathematical properties into stunning visual art.

**Conservation ratio** — how smoothly attributes flow across a graph's structure — drives color harmony, layer coherence, and visual tension.

## Gallery

### 1. Spectral Mandalas 🌀
Eigenvectors define angles and radii; eigenvalues control mandala layers. Conservation ratio governs color harmony.

- Erdős-Rényi Constellation
- Barabási-Albert Nebula
- Small World Chakra
- Grid Lattice Bloom
- Western / Gamelan / Indian / Jazz / African Tradition Mandalas

### 2. Eigenvalue Landscapes 🏔️
Eigenvector 2 × Eigenvector 3 coordinates form the terrain. Attribute values become elevation. Conservation ratio maps to color.

- Scale-Free Topology
- Small-World Terrain
- Lattice Prairie
- Gamelan / Jazz Spectral Terrain

### 3. Laplacian Flow Fields 🌊
Graph nodes become attractors. Laplacian eigenvectors define flow direction. Conservation equals flow coherence.

- Laplacian Vortex
- Scale-Free Currents
- Random Walk Dreams
- Western / Gamelan Tradition Flows

### 4. Tradition Portraits 🎭
Each musical tradition's Laplacian eigenvalues define a unique abstract shape.

- **Western** — Structured geometric polygons
- **Gamelan** — Layered organic curves
- **Indian** — Spiral raga complexity
- **Jazz** — Improvisational chaos
- **African** — Polyrhythmic bold patterns

### 5. Conservation Heat Maps 🌡️
Multi-scale conservation ratio mapped to color gradients. Shows the transition from well-conserved (smooth) to anomalous (rough).

- Scale-Free / Small-World / Random / Lattice Conservation

## Installation

```bash
pip install numpy scipy matplotlib pillow
```

## Usage

```bash
python generate.py
```

All output saved to `output/` as PNG files.

## Mathematical Foundation

- **Normalized Laplacian**: L = I − D^{−1/2} A D^{−1/2}
- **Conservation Ratio**: σ(x) = 1 / (1 + x^T L x / x^T x)
- **Rayleigh Quotient**: Measures smoothness of attribute x on graph structure
- **Spectral Embedding**: Eigenvectors map graph to visual space

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
