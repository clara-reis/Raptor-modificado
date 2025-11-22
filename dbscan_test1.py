import os
import logging

from datasets import load_dataset
import tiktoken
from utils import split_text # do RAPTOR
from sentence_transformers import SentenceTransformer
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from collections import Counter
from EmbeddingModels import SBertEmbeddingModel
from SummarizationModels import GPT3TurboSummarizationModel
from tree_builder import TreeBuilder
from cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from cluster_utils import global_cluster_embeddings,local_cluster_embeddings
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO)
#IMPORTAR DATASET QASPER

def load_qasper(split="validation"):
    return load_dataset("allenai/qasper", split=split)

ds = load_dataset("allenai/qasper", split="validation")
print(ds[0].keys())
ex=ds[0] #seleciona o primeiro paper 

#Reconstrói o texto do paper a partir de full_text
paper_text = "\n\n".join([" ".join(p) for p in ex["full_text"]["paragraphs"]])


# Usa a função split_text do RAPTOR 

tokenizer = tiktoken.get_encoding("cl100k_base") 
chunks = split_text(paper_text, tokenizer, max_tokens=100)
# Exibe os dois primeiros chunks
print(f"Total de chunks gerados: {len(chunks)}\n")
print("Chunk 1:\n", chunks[0], "\n")
print("Chunk 2:\n", chunks[1])

# Converter chunks em embeddings
#Embeddings 
emb_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1")
embeddings = emb_model.encode(chunks, show_progress_bar=True)
embeddings = np.asarray(embeddings)
#dim = embeddings.shape[1]
N, dim = embeddings.shape
print(f"Shape das embeddings: {embeddings.shape}  (N={N}, dim={dim})")
print(len(embeddings))


# Definir o modelo de embedding
embedding_models = {
    "mpnet": SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")
}

#Redução da dimensionalidade UMAP
#A função global_cluster_embeddings é usada para aplicar redução de dimensionalidade nas embeddings dos chunks de texto com o UMAP
reduction_dimension=10

reduced_embeddings = global_cluster_embeddings(embeddings,reduction_dimension, n_neighbors=None,metric="cosine")
Nr, dimr= reduced_embeddings.shape
print(f"Shape das embeddings: {reduced_embeddings.shape}  (N={Nr}, dim={dimr})")

'''
pca = PCA(n_components=2)
embedding_2d = pca.fit_transform(reduced_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=10)
plt.title("PCA 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
'''

# Function to plot k-distance graph
def plot_k_distance_graph(X, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances[:, k-1])
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.title('K-distance Graph')
    plt.grid()
    plt.show()
# Plot k-distance graph
plot_k_distance_graph(reduced_embeddings, k=5)

# Perform DBSCAN clustering
epsilon = 0.75  # Chosen based on k-distance graph
min_samples = 5  # 2 * num_features (2D data)
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(reduced_embeddings)

# Print number of clusters and noise points
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {n_noise}')