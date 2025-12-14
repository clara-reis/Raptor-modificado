import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from minisom import MiniSom

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from tree_structures import Node

# Import necessary methods from other modules
from utils import get_embeddings

# Set a random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


#global_cluster_embeddings e  local_cluster_embeddings fazem redução de dimensionalidade com UMAP
#primeiro reduzir para um espaço de baixa dimensão 
#depois rodar GMM nesse espaço em duas etapas: global e local 

def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    n_samples = len(embeddings)

    # Para poucos pontos, UMAP/spectral pode falhar (k >= N); nesse caso, não reduz.
    if n_samples <= 3:
        return embeddings

    if n_neighbors is None:
        n_neighbors = int((n_samples - 1) ** 0.5)

    # Garante 2 <= n_neighbors <= n_samples - 1
    n_neighbors = max(2, min(n_neighbors, n_samples - 1))

    # Garante que n_components < n_samples para evitar k >= N no espectral
    if dim >= n_samples:
        dim = max(1, n_samples - 1)

    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric,
        init="random",  # evita spectral layout e chamada a eigsh com k>=N
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    n_samples = len(embeddings)

    if n_samples <= 3:
        return embeddings

    # Garante 2 <= num_neighbors <= n_samples - 1
    num_neighbors = max(2, min(num_neighbors, n_samples - 1))

    if dim >= n_samples:
        dim = max(1, n_samples - 1)

    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors,
        n_components=dim,
        metric=metric,
        init="random",
    ).fit_transform(embeddings)
    return reduced_embeddings


#Tenta descobrir o número ótimo de componentes do GMM com base no critério BIC.
def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters

#Treina um GMM com n_clusters componentes.
#gm.predict_proba(embeddings) devolve, para cada ponto, um vetor de probabilidades p(cluster=k | x).
#np.where(prob > threshold)[0] pega todos os clusters com probabilidade > limiar.
def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters #all_local_clusters[i] é um np.array contendo os IDs de cluster aos quais o ponto i pertence.


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass

class ClusteringAlgorithm_som(ABC):
    @abstractmethod
    def perform_clustering_som(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        som_x: int = 5,
        som_y: int = 5,
        som_iterations: int = 1500,
        verbose: bool = False,
    ) -> List[List[Node]]:
        """
        Interface específica para algoritmos baseados em SOM.
        """
        pass

class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
        # Get the embeddings from the nodes
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

        # Perform the clustering -> clusters aqui é exatamente o all_local_clusters retornado pela função anterior;
        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes, embedding_model_name, max_length_in_cluster
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters



##########
#Versão SOM
def som_cluster(
    embeddings: np.ndarray,
    som_x: int = 5,
    som_y: int = 5,
    num_iterations: int = 1500,
    random_order: bool = True,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Clusteriza embeddings usando um SOM de tamanho som_x x som_y.

    Retorna uma lista `clusters` em que clusters[i] é um np.array
    com um único inteiro: o ID do neurônio vencedor (cluster) 
    para o embedding i. O formato é compatível com o usado pelo 
    RAPTOR_Clustering.perform_clustering.
    """
    input_len = embeddings.shape[1]

    # Cria o SOM
    som = MiniSom(
        som_x,
        som_y,
        input_len,
        sigma=1.0,
        learning_rate=0.5,
        random_seed=RANDOM_SEED,
    )

    # Treina o SOM
    if random_order:
        som.train_random(embeddings, num_iterations)
    else:
        som.train(embeddings, num_iterations)

    # Atribui um cluster (BMU) para cada embedding
    clusters: List[np.ndarray] = []
    for x in embeddings:
        i, j = som.winner(x)
        cid = i * som_y + j  # índice linear do neurônio (0 .. som_x*som_y-1)
        clusters.append(np.array([cid], dtype=int))

    if verbose:
        logging.info(f"Total de neurônios SOM (clusters potenciais): {som_x * som_y}")
        logging.info(
            f"Clusters efetivamente usados: {len(np.unique(np.concatenate(clusters)))}"
        )

    return clusters 

class SOM_Clustering(ClusteringAlgorithm_som):
    def perform_clustering_som(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        som_x: int = 5,
        som_y: int = 5,
        som_iterations: int = 1500,
        verbose: bool = False,
    ) -> List[List[Node]]:
        # Get the embeddings from the nodes
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

        # Perform the clustering com SOM em vez de GMM/UMAP
        clusters = som_cluster(
            embeddings,
            som_x=som_x,
            som_y=som_y,
            num_iterations=som_iterations,
            random_order=True,
            verbose=verbose,
        )

        node_clusters: List[List[Node]] = []

        # Resto do código permanece IGUAL
        for label in np.unique(np.concatenate(clusters)):
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]

            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    SOM_Clustering.perform_clustering_som(
                        cluster_nodes,
                        embedding_model_name,
                        max_length_in_cluster=max_length_in_cluster,
                        tokenizer=tokenizer,
                        som_x=som_x,
                        som_y=som_y,
                        som_iterations=som_iterations,
                        verbose=verbose,
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters

class RAPTOR_DBSCAN_Clustering(ClusteringAlgorithm):
    @staticmethod
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        eps: float = 0.75,
        min_samples: int = 5,
        verbose: bool = False,
    ) -> List[List[Node]]:
        """Cluster nodes using UMAP reduction + DBSCAN.

        The extra "threshold" and som_* arguments are accepted only for API
        compatibility with RAPTOR_Clustering and are ignored by DBSCAN.
        """

        if not nodes:
            return []

        # Get embeddings matrix (N, D)
        embeddings = np.array([
            node.embeddings[embedding_model_name] for node in nodes
        ])

        if len(embeddings) == 1:
            return [[nodes[0]]]

        # Reduce dimensionality with UMAP before DBSCAN
        effective_dim = max(2, min(reduction_dimension, len(embeddings) - 1))
        reduced_embeddings = global_cluster_embeddings(
            embeddings,
            effective_dim,
            metric="cosine",
        )

        # Run DBSCAN in the reduced space
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(reduced_embeddings)

        unique_labels = set(labels)
        node_clusters: List[List[Node]] = []

        for label in unique_labels:
            indices = np.where(labels == label)[0]

            # Noise points (-1) become single-node clusters
            if label == -1:
                for idx in indices:
                    node_clusters.append([nodes[idx]])
                continue

            cluster_nodes = [nodes[i] for i in indices]

            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            total_length = sum(
                len(tokenizer.encode(node.text)) for node in cluster_nodes
            )

            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"[DBSCAN] Reclustering cluster with {len(cluster_nodes)} nodes, total_length={total_length}"
                    )
                node_clusters.extend(
                    RAPTOR_DBSCAN_Clustering.perform_clustering(
                        cluster_nodes,
                        embedding_model_name,
                        max_length_in_cluster=max_length_in_cluster,
                        tokenizer=tokenizer,
                        reduction_dimension=reduction_dimension,
                        threshold=threshold,
                        eps=eps,
                        min_samples=min_samples,
                        verbose=verbose,
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
