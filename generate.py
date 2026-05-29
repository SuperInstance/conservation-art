#!/usr/bin/env python3
"""Generate all conservation-aware generative art pieces."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.mandala import generate_mandala_gallery
from src.landscapes import generate_landscape_gallery
from src.flow_fields import generate_flow_gallery
from src.portraits import generate_portrait_gallery
from src.heatmaps import generate_heatmap_gallery

if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CONSERVATION-AWARE GENERATIVE ART")
    print("=" * 60)
    
    print("\n🌀 Spectral Mandalas...")
    generate_mandala_gallery(output_dir)
    
    print("\n🏔️  Eigenvalue Landscapes...")
    generate_landscape_gallery(output_dir)
    
    print("\n🌊 Laplacian Flow Fields...")
    generate_flow_gallery(output_dir)
    
    print("\n🎭 Tradition Portraits...")
    generate_portrait_gallery(output_dir)
    
    print("\n🌡️  Conservation Heat Maps...")
    generate_heatmap_gallery(output_dir)
    
    print("\n" + "=" * 60)
    print(f"✨ All pieces saved to: {output_dir}")
    print("=" * 60)
