#!/usr/bin/env python3
"""Example usage of enhanced embedding visualizer."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Example: Basic clustering visualization
def example_clustering():
    """Demonstrate clustering of text embeddings."""
    texts = [
        "Machine learning algorithms learn from data",
        "Deep learning uses neural networks",
        "Supervised learning requires labeled data",
        "Python is a programming language",
        "JavaScript runs in web browsers",
        "Java is used for enterprise apps",
    ]
    
    print("Texts for clustering:")
    for i, t in enumerate(texts):
        print(f"  {i+1}. {t}")
    
    # In actual usage:
    # from visualize_enhanced import EmbeddingVisualizerEnhanced
    # viz = EmbeddingVisualizerEnhanced()
    # viz.add_texts(texts)
    # viz.compute_embeddings()
    # viz.find_clusters(n_clusters=2)
    # viz.create_interactive_visualization("output.html", color_by="cluster")

if __name__ == "__main__":
    example_clustering()
