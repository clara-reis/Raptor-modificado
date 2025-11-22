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


logging.basicConfig(level=logging.INFO)



ds = load_dataset("allenai/qasper", split="validation")
print(ds[0].keys())
example=ds[0] #seleciona o primeiro paper 
# --------- 1) Utilitário: montar texto de um paper do QASPER ---------
def qasper_example_to_text(example):
    paper_text = "\n\n".join([" ".join(p) for p in example["full_text"]["paragraphs"]])
    return paper_text 


# --------- 2) Função para plotar a árvore em estilo RAPTOR ---------
def plot_tree(tree, save_path=None):
    """
    Desenha a árvore RAPTOR em 2D, com camadas horizontais
    e nós conectados pelos filhos.

    O resultado lembra a figura do paper.
    """
    # tree.layer_to_nodes: dict layer -> [Node]
    layer_to_nodes = tree.layer_to_nodes
    max_layer = max(layer_to_nodes.keys())

    # posição (x, y) de cada nó
    positions = {}

    for layer, nodes in layer_to_nodes.items():
        y = max_layer - layer  # camada 0 embaixo, raiz em cima
        n = len(nodes)
        if n == 0:
            continue
        # distribui x de forma espaçada
        xs = np.linspace(0.05, 0.95, n)
        for x, node in zip(xs, nodes):
            positions[node.index] = (x, y)

    fig, ax = plt.subplots(figsize=(12, 3 + 1.2 * max_layer))

    # desenhar arestas (pai -> filhos)
    # dependendo da implementação do Tree, os nós podem estar em tree.nodes ou tree.all_nodes;
    # aqui assumo tree.nodes; se der erro, troque para tree.all_nodes.
    all_nodes_dict = getattr(tree, "nodes", None)
    if all_nodes_dict is None:
        all_nodes_dict = getattr(tree, "all_nodes", None)

    for node in all_nodes_dict.values():
        x, y = positions[node.index]
        for child_idx in node.children:
            if child_idx not in positions:
                continue
            cx, cy = positions[child_idx]
            ax.plot([x, cx], [y, cy], color="lightgray", linewidth=0.7, zorder=1)

    # desenhar nós
    for node in all_nodes_dict.values():
        x, y = positions[node.index]
        ax.scatter(x, y, s=80, color="#607fb0", edgecolor="k", linewidths=0.5, zorder=2)

    # destacar raízes em laranja
    for root in tree.root_nodes:
        x, y = positions[root.index]
        ax.scatter(x, y, s=90, color="#f2a65a", edgecolor="k", linewidths=0.7, zorder=3)

    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# --------- 3) Pipeline completo: QASPER -> árvore -> figura ---------
def main():
    # 1) Carregar QASPER (validation) e escolher um paper
    print("Carregando QASPER (validation)...")
    ds = load_dataset("allenai/qasper", split="validation")

    ex_idx = 0  # você pode mudar o índice para outro paper
    example = ds[ex_idx]
    article_text = qasper_example_to_text(example)

    print(f"\nPaper index: {ex_idx}")
    print("ID:", example.get("id"))
    print("Title:", example.get("title"))
    print("Tamanho do texto (caracteres):", len(article_text))

    # 2) Tokenizer (mesmo que você usou para split_text)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # 3) Modelo de embedding local (SBERT / MPNet)
    embedding_models = {
        "SBERT": SBertEmbeddingModel(
            "sentence-transformers/multi-qa-mpnet-base-cos-v1"
            # ou "sentence-transformers/all-mpnet-base-v2"
        )
    }

    # 4) Modelo de sumarização (LLaMA 3.2 via Ollama/OpenAI compatível)
    summarizer = GPT3TurboSummarizationModel(model="llama3.2:3b")

    # 5) Configuração do ClusterTreeBuilder
    config = ClusterTreeConfig(
        tokenizer=tokenizer,
        max_tokens=100,                  # tamanho dos chunks em tokens
        num_layers=5,                    # limite máximo de camadas
        threshold=0.5,
        top_k=5,
        selection_mode="top_k",
        summarization_length=120,        # tamanho do resumo em cada nó
        summarization_model=summarizer,
        embedding_models=embedding_models,
        cluster_embedding_model="SBERT",
        # se você adicionou som_x, som_y, som_iterations na config, pode passar aqui;
        # se não, a versão do RAPTOR_Clustering usa os defaults (5x5, 1500 iterações)
    )

    # 6) Construir árvore com SOM (isso chama internamente o Raptor_Clustering com SOM)
    tree_builder = ClusterTreeBuilder(config)

    print("\nConstruindo árvore (SOM + LLaMA)...")
    tree = tree_builder.build_from_text(article_text, use_multithreading=False)

    # 7) Resumo rápido da árvore
    print("\n--- Estrutura da árvore ---")
    print("Número de folhas (chunks iniciais):", len(tree.leaf_nodes))
    print("Número de camadas:", tree.num_layers)
    for layer, nodes in tree.layer_to_nodes.items():
        print(f"  Camada {layer}: {len(nodes)} nós")

    # 8) Mostrar a(s) raiz(es)
    print("\n--- Nó(s) raiz ---")

    roots = tree.root_nodes

'''
    # Caso seja um dict {idx: Node}
    if isinstance(roots, dict):
        for root_idx, root_node in roots.items():
            print(f"Root index: {root_idx}")
            prev = root_node.text.replace("\n", " ")
            print(prev[:400], "...")
            print("-" * 80)

    # (Opcional) se em algum cenário for lista de Nodes:
    else:
        for root in roots:
            print(f"Root index: {root.index}")
            prev = root.text.replace("\n", " ")
            print(prev[:400], "...")
            print("-" * 80)

    # 9) Plotar e salvar figura da árvore
    print("\nGerando figura da árvore...")
    os.makedirs("figuras", exist_ok=True)
    save_path = os.path.join("figuras", f"qasper_tree_som_paper_{ex_idx}.png")
    plot_tree(tree, save_path=save_path)
    print("Figura salva em:", save_path)
'''

if __name__ == "__main__":
    main()