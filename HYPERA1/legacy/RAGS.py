from skimage import io, graph
import networkx as nx
import matplotlib.pyplot as plt

# Load a labeled mask
labeled_mask = io.imread('path/to/mask.tiff')

# Create a Region Adjacency Graph
rag = graph.RAG(labeled_mask)

# Visualize the RAG
plt.figure(figsize=(10, 10))
graph.show_rag(labeled_mask, rag, edge_cmap='viridis', edge_width=1.5)
plt.title('Region Adjacency Graph')
plt.axis('off')
plt.tight_layout()
plt.savefig('rag_visualization.png')