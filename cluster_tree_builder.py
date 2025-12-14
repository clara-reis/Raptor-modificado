import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set

from cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering, ClusteringAlgorithm_som, SOM_Clustering
from tree_builder import TreeBuilder, TreeBuilderConfig
from tree_structures import Node, Tree
from utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

'''
script responsavel por pegar os nós da camada, clusterizar, sumarizar, criar pais, subir de camada.
'''

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

##### ORIGINAL GMM ###########
class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,  # Default to RAPTOR clustering
        clustering_params={},  # Pass additional params as a dict
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            node_texts = get_text(cluster)

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            __, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.num_layers):

            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )

            lock = Lock()

            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes
##############################

####SOM ###############

class ClusterTreeConfig_som(TreeBuilderConfig):
    def __init__(
        self,
        clustering_algorithm=SOM_Clustering,  
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clustering_algorithm = clustering_algorithm
        

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        SOM Clustering Algorithm: {self.clustering_algorithm.__name__}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder_som(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig_som):
            raise ValueError("config must be an instance of ClusterTreeConfig_som")
        self.clustering_algorithm = config.clustering_algorithm
        

        logging.info(
            f"Successfully initialized ClusterTreeBuilder_som with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node], #todos os nós criados até agora.
        layer_to_nodes: Dict[int, List[Node]], #mapeia layer → [Nodes]
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            node_texts = get_text(cluster) #concatena os textos dos nós no cluster.

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            ) #chama o modelo de sumarização (agora LLaMA local via SummarizationModels.py adaptado).

            logging.info(
                f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            __, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node #Guarda o nó pai no dicionário new_level_nodes da próxima camada.
        #Loop sobre as camadas
        for layer in range(self.num_layers):

            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes) #node_list_current_layer vira uma lista simples de Nodes (sem dicionário)
            
            # Define o tamanho do grid SOM com base na heurística
            num_chunks = len(node_list_current_layer)
            lattice_size = int((5 * (num_chunks ** 0.5)) ** 0.5)
            lattice_size = max(lattice_size - 1, 1)  # Evita 0
            som_x = som_y = lattice_size

            logging.info(f"Layer {layer}: {num_chunks} chunks → SOM grid {som_x}x{som_y}")
          
            #Mudar o critério de parada da árvore
            if len(node_list_current_layer) <= 1 or lattice_size == 1 :
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: only one node left. Total layers in tree: {layer}"
                )
                break

            #Ajustar a chamada do algoritmo de clusterização
            clusters = self.clustering_algorithm.perform_clustering_som(
                node_list_current_layer,              # nodes
                self.cluster_embedding_model,         # embedding_model_name
                tokenizer=self.tokenizer,             # usa o mesmo tokenizer do TreeBuilder
                som_x=som_x,
                som_y=som_y,
                som_iterations=1500,
                verbose=False,
            )
            
            # adicionar critério de parada por estabilidade
            # Verifica se a clusterização não está progredindo (estagnou)
            if len(clusters) >= len(current_level_nodes):
                logging.info(f"Stopping: clustering stalled (clusters={len(clusters)}, nodes={len(current_level_nodes)})")
                break

            lock = Lock()

            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes




