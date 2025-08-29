import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
import os

# Paths
input_file = "/home/sjet/iwazolab/NLP-Scam-Detection/data/processed/scam_preprocessed.csv"
output_dir = "/home/sjet/iwazolab/NLP-Scam-Detection/outputs/assets"
os.makedirs(output_dir, exist_ok=True)

# Load preprocessed dataset
df = pd.read_csv(input_file)
texts = df['cleaned_text'].dropna().astype(str)

# Extract top 30 quadgrams
vectorizer = CountVectorizer(ngram_range=(4,4))
X = vectorizer.fit_transform(texts)
quadgrams = vectorizer.get_feature_names_out()
counts = X.sum(axis=0).A1
quadgram_freq = sorted(zip(quadgrams, counts), key=lambda x: x[1], reverse=True)[:30]

# Build directed graph
G = nx.DiGraph()
for quadgram, freq in quadgram_freq:
    words = quadgram.split()
    if len(words) == 4:
        w1, w2, w3, w4 = words
        # Connect sequentially: w1 -> w2 -> w3 -> w4
        G.add_edge(w1, w2, weight=freq)
        G.add_edge(w2, w3, weight=freq)
        G.add_edge(w3, w4, weight=freq)

# Use Fruchterman-Reingold layout
plt.figure(figsize=(15, 12))
pos = nx.fruchterman_reingold_layout(G, k=5, iterations=400)

# Node size by degree
node_sizes = [200 + 300 * G.degree(n) for n in G.nodes()]

# Edge width by normalized weight
edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]
max_w = max(weights) if weights else 1
edge_widths = [0.5 + 1.0 * (w / max_w) for w in weights]

# Draw nodes & labels
nx.draw_networkx_nodes(G, pos, node_color="#FFD580", node_size=node_sizes, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="normal")

# Draw edges
nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths, edge_color="#444444", arrowsize=10)

plt.title("Top 30 Quadgram Network", fontsize=18)
plt.axis("off")
plt.tight_layout()

# Save as high-quality PNG
output_path = os.path.join(output_dir, "quadgram_network.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✅ Quadgram network saved to: {output_path}")
