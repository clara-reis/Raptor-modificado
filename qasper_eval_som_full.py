import argparse
import logging
import os
from typing import List, Tuple

import json

import numpy as np
from datasets import load_dataset
from openai import OpenAI
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from EmbeddingModels import SBertEmbeddingModel
from SummarizationModels import GPT3TurboSummarizationModel, GeminiSummarizationModel
from cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig, ClusterTreeConfig_som, ClusterTreeBuilder_som
from cluster_utils import RAPTOR_Clustering, RAPTOR_DBSCAN_Clustering, SOM_Clustering #Importar os 3 algoritmos
from utils import (
    get_embeddings,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
    get_text,
)

try:
    import google.generativeai as genai
except ImportError:
    genai = None


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

############
#Salvar progresso em um json
progress_file = "progress.json"
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        completed_papers = set(json.load(f))
else:
    completed_papers = set()
###################



def qasper_example_to_text(example) -> str:
    paragraphs = example["full_text"]["paragraphs"]
    return "\n\n".join([" ".join(p) for p in paragraphs])


def build_tree_for_paper(
    example,
    tokenizer,
    summarizer_backend: str = "ollama",
    summarizer_model_name: str = "llama3.2:3b",
    cluster_backend: str = "full",
):
    logging.info(f"DEBUG cluster_backend recebido = '{cluster_backend}'")
    article_text = qasper_example_to_text(example)

    embedding_models = {
        "SBERT": SBertEmbeddingModel(
            "sentence-transformers/multi-qa-mpnet-base-cos-v1"
        )
    }

    if summarizer_backend == "gemini":
        summarizer = GeminiSummarizationModel(model=summarizer_model_name)
    else:
        summarizer = GPT3TurboSummarizationModel(model=summarizer_model_name)

    if cluster_backend == "som":
        # SOM → usa classes _som
        clustering_algorithm = SOM_Clustering
        config = ClusterTreeConfig_som(
            clustering_algorithm=clustering_algorithm,
            tokenizer=tokenizer,
            max_tokens=100,
            num_layers=5,
            threshold=0.5,
            top_k=5,
            selection_mode="top_k",
            summarization_length=120,
            summarization_model=summarizer,
            embedding_models=embedding_models,
            cluster_embedding_model="SBERT",
        )
        tree_builder = ClusterTreeBuilder_som(config)
        clustering_algorithm = SOM_Clustering

    elif cluster_backend == "dbscan":
        # DBSCAN → builder genérico
        clustering_algorithm = RAPTOR_DBSCAN_Clustering
        config = ClusterTreeConfig(
            clustering_algorithm=clustering_algorithm,
            tokenizer=tokenizer,
            max_tokens=100,
            num_layers=5,
            threshold=0.5,
            top_k=5,
            selection_mode="top_k",
            summarization_length=120,
            summarization_model=summarizer,
            embedding_models=embedding_models,
            cluster_embedding_model="SBERT",
        )
        tree_builder = ClusterTreeBuilder(config)

    else:
        # GMM (RAPTOR original)
        clustering_algorithm = RAPTOR_Clustering
        config = ClusterTreeConfig(
            clustering_algorithm=clustering_algorithm,
            tokenizer=tokenizer,
            max_tokens=100,
            num_layers=5,
            threshold=0.5,
            top_k=5,
            selection_mode="top_k",
            summarization_length=120,
            summarization_model=summarizer,
            embedding_models=embedding_models,
            cluster_embedding_model="SBERT",
            )
        tree_builder = ClusterTreeBuilder(config)

    tree = tree_builder.build_from_text(article_text, use_multithreading=False)
    return tree_builder, tree


def retrieve_context(
    tree_builder: ClusterTreeBuilder,
    tree,
    query: str,
    top_k: int = 5,
) -> Tuple[str, List[int]]:
    all_nodes_dict = tree.all_nodes
    node_list = list(all_nodes_dict.values())

    node_embeddings = get_embeddings(node_list, tree_builder.cluster_embedding_model)
    query_embedding = tree_builder.create_embedding(query)
    distances = distances_from_embeddings(query_embedding, node_embeddings, "cosine")
    indices = indices_of_nearest_neighbors_from_distances(distances)

    selected_indices = indices[:top_k]
    selected_nodes = [node_list[i] for i in selected_indices]
    context = get_text(selected_nodes)

    return context, [node.index for node in selected_nodes]


def answer_question_with_llama(
    client: OpenAI,
    model: str,
    context: str,
    question: str,
    max_tokens: int = 150,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a question answering assistant. Answer concisely based only on the provided context.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def answer_question_with_gemini(
    context: str,
    question: str,
    model: str = "gemini-2.0-flash-lite",
    max_tokens: int = 150,
) -> str:

    if genai is None:
        raise RuntimeError("google-generativeai is not installed in this environment.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)

    prompt = (
        "You are a question answering assistant. Answer concisely based only on the provided context.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    response = gmodel.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens},
    )

    return response.text.strip()

##funções de avaliação (F1)
def normalize_text(s: str) -> List[str]:
    import re

    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [t for t in s.split() if t]
    return tokens

#Sobreposição de token
def f1_score(prediction: str, ground_truths: List[str]) -> float:
    import math

    if not ground_truths:
        return 0.0

    pred_tokens = normalize_text(prediction)
    if len(pred_tokens) == 0:
        return 0.0

    def f1_single(gt: str) -> float:
        gt_tokens = normalize_text(gt)
        if len(gt_tokens) == 0:
            return 0.0
        common = {}
        for t in pred_tokens:
            common[t] = common.get(t, 0) + 1
        num_same = 0
        for t in gt_tokens:
            if common.get(t, 0) > 0:
                num_same += 1
                common[t] -= 1
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    return max(f1_single(gt) for gt in ground_truths)


def extract_answer_texts(ans_item) -> List[str]:
    gold: List[str] = []

    if not isinstance(ans_item, dict):
        return gold

    annotations = ans_item.get("answer", [])
    if not isinstance(annotations, list):
        return gold

    for ann in annotations:
        if not isinstance(ann, dict):
            continue

        free_form = ann.get("free_form_answer", "")
        if isinstance(free_form, str) and free_form.strip():
            gold.append(free_form.strip())

        spans = ann.get("extractive_spans", [])
        if isinstance(spans, list):
            for span in spans:
                if isinstance(span, str) and span.strip():
                    gold.append(span.strip())

        yn = ann.get("yes_no", None)
        if yn is not None:
            gold.append("yes" if yn else "no")

    # remove duplicatas preservando ordem
    seen = set()
    deduped: List[str] = []
    for text in gold:
        if text not in seen:
            seen.add(text)
            deduped.append(text)

    return deduped


def extract_gold_answer_texts(ans_item):
    if isinstance(ans_item, dict):
        if "free_form_answer" in ans_item:
            return [ans_item["free_form_answer"]]
        elif "extractive_spans" in ans_item:
            return [span["text"] for span in ans_item["extractive_spans"]]
        elif "yes_no" in ans_item:
            return [ans_item["yes_no"]]
    return []


def infer_question_type(ans_item) -> str:
    """Infer QASPER question type from its answer structure.

    Categories (mutually exclusive, priority order inspired by QASPER paper):
    - boolean: any annotation with yes_no not None
    - extractive: any annotation with non-empty extractive_spans
    - abstractive: any annotation with non-empty free_form_answer
    - none: otherwise (unanswerable / no-annotation)
    """

    if not isinstance(ans_item, dict):
        return "none"

    annotations = ans_item.get("answer", [])
    if not isinstance(annotations, list) or len(annotations) == 0:
        return "none"

    has_boolean = False
    has_extractive = False
    has_abstractive = False

    for ann in annotations:
        if not isinstance(ann, dict):
            continue

        if ann.get("yes_no", None) is not None:
            has_boolean = True

        spans = ann.get("extractive_spans", [])
        if isinstance(spans, list) and any(isinstance(s, str) and s.strip() for s in spans):
            has_extractive = True

        free_form = ann.get("free_form_answer", "")
        if isinstance(free_form, str) and free_form.strip():
            has_abstractive = True

    if has_boolean:
        return "boolean"
    if has_extractive:
        return "extractive"
    if has_abstractive:
        return "abstractive"
    return "none"


def evaluate_qasper_subset(
    num_papers: int = 1,
    num_questions_per_paper: int = 5,
    model_name: str = "llama3.2:3b",
    split: str = "validation",
    backend: str = "ollama",
    cluster_backend: str = "dbscan",
    output_file: str = None,
) -> None:
    logging.info(
        f"Evaluating QASPER subset: num_papers={num_papers}, num_questions_per_paper={num_questions_per_paper}, model={model_name}"
    )

    ds = load_dataset("allenai/qasper", split=split)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    client = None
    if backend == "ollama":
        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        )

    f1_scores: List[float] = []
    num_eval = 0

    # F1 acumulado por tipo de pergunta
    type_f1: dict = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    type_counts: dict = {k: 0 for k in type_f1.keys()}

    tree_stats_lines: List[str] = []

    max_papers = len(ds) if num_papers is None or num_papers <= 0 else min(num_papers, len(ds))

    for paper_idx in range(max_papers):
        example = ds[paper_idx]

        '''
        ############
        #Salvar progresso em um json
        paper_id = example.get("id")
        if paper_id in completed_papers:
            logging.info(f"Skipping already completed paper: {paper_id}")
            continue
        ###################
        '''

        qas = example.get("qas", None)
        if not isinstance(qas, dict):
            continue

        questions_list = qas.get("question", [])
        answers_list = qas.get("answers", [])

        if not questions_list or not answers_list:
            continue

        num_qas = min(len(questions_list), len(answers_list))

        logging.info(
            f"Building tree for paper {paper_idx} with id={example.get('id')} title={example.get('title')} and {num_qas} questions"
        )

        tree_builder = None
        tree = None
        full_article_text = None

        if cluster_backend == "full":
            # Baseline: usar o artigo inteiro como contexto, sem RAPTOR / clusterização
            logging.info("Usando baseline FULL: documento inteiro como contexto (sem RAPTOR)")
            full_article_text = qasper_example_to_text(example)
        else:
            # Construir árvore RAPTOR normalmente (som, gmm ou dbscan)
            tree_builder, tree = build_tree_for_paper(
                example,
                tokenizer,
                summarizer_backend=backend,
                summarizer_model_name=model_name,
                cluster_backend=cluster_backend,
            )
        # Opcional: registrar estatísticas básicas da árvore para este paper
        if output_file is not None:
            try:
                num_leaves = len(tree.leaf_nodes)
                num_layers = tree.num_layers
                layer_sizes = []
                for layer_idx, nodes_layer in sorted(tree.layer_to_nodes.items()):
                    layer_sizes.append(f"  Camada {layer_idx}: {len(nodes_layer)} nós")
                header = [
                    f"Paper {paper_idx}",
                    f"ID: {example.get('id')}",
                    f"Title: {example.get('title')}",
                    f"Número de folhas: {num_leaves}",
                    f"Número de camadas: {num_layers}",
                ]
                tree_stats_lines.append("\n".join(header + layer_sizes))
            except Exception as e:
                logging.warning(f"Falha ao coletar estatísticas da árvore para paper {paper_idx}: {e}")

        max_qas = num_qas if num_questions_per_paper is None or num_questions_per_paper <= 0 else min(num_qas, num_questions_per_paper)

        for qa_idx in range(max_qas):
            question = questions_list[qa_idx]
            ans_item = answers_list[qa_idx]

            q_type = infer_question_type(ans_item)
            #answer_texts = extract_answer_texts(ans_item)
            #evitar avaliar perguntas sem gold answers de forma mais robusta
            answer_texts = [a.strip() for a in extract_answer_texts(ans_item) if a.strip()]
            if not question.strip() or not answer_texts:
                continue

            logging.info(
                f"Paper {paper_idx}, QA {qa_idx}: question='{question[:80]}...', num_gold_answers={len(answer_texts)}"
            )


             # Escolher contexto de acordo com o backend
            if cluster_backend == "full":
                context = full_article_text
                node_indices = []
            else:
                context, node_indices = retrieve_context(tree_builder, tree, question, top_k=5)

            try:
                if backend == "gemini":
                    prediction = answer_question_with_gemini(
                        context=context,
                        question=question,
                        model=model_name,
                    )
                else:
                    prediction = answer_question_with_llama(
                        client=client,
                        model=model_name,
                        context=context,
                        question=question,
                    )
            except Exception as e:
                logging.warning(
                    f"Skipping QA due to backend error (backend={backend}): {e}"
                )
                continue

            score = f1_score(prediction, answer_texts)
            f1_scores.append(score)
            num_eval += 1

            ############################
            # Novo bloco para exibir pergunta, resposta e gold answer mais próxima
            best_gold = None
            best_score = -1.0
            bert_best_score = 0.0
            for gt in answer_texts:
                s = f1_score(prediction, [gt])
                if s > best_score:
                    best_score = s
                    best_gold = gt

            # Garantia extra para evitar None
            if best_gold is None and answer_texts:
                best_gold = answer_texts[0]
                ##################################
            '''
            print("\n===== Avaliação Detalhada =====")
            print(f"Pergunta: {question}")
            print(f"Resposta gerada: {prediction}")
            print(f"Melhor resposta ouro: {best_gold}")
            print(f"F1 entre gerada e melhor ouro: {best_score:.4f}")
            print("================================\n")
            '''
           
            ############################

            if q_type not in type_f1:
                q_type = "none"
            type_f1[q_type].append(score)
            type_counts[q_type] += 1

            logging.info(f"F1 for this QA: {score:.4f}")
    '''
    completed_papers.add(paper_id)
    with open(progress_file, "w") as f:
        json.dump(list(completed_papers), f)
    '''


    if num_eval == 0:
        logging.info("No QASPER questions were evaluated.")
        return

    avg_f1 = float(np.mean(f1_scores))
    logging.info(f"Evaluated {num_eval} QASPER questions. Average F1 = {avg_f1:.4f}")

    # F1 médio por tipo de pergunta
    type_avg_f1 = {}
    for t, scores in type_f1.items():
        if scores:
            type_avg_f1[t] = float(np.mean(scores))
        else:
            type_avg_f1[t] = float("nan")

    # Se um arquivo de saída foi especificado, salvar métricas e estatísticas da árvore
    if output_file is not None:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("# QASPER RAPTOR Evaluation\n\n")
                f.write(f"Split: {split}\n")
                f.write(f"Backend LLM: {backend}\n")
                f.write(f"Modelo: {model_name}\n")
                f.write(f"Cluster backend: {cluster_backend}\n")
                f.write(f"Perguntas avaliadas: {num_eval}\n")
                f.write(f"Answer F1 médio (global): {avg_f1:.4f}\n")

                f.write("\n## Answer F1 por tipo de pergunta\n\n")
                for t in ["extractive", "abstractive", "boolean", "none"]:
                    count = type_counts.get(t, 0)
                    avg = type_avg_f1.get(t, float("nan"))
                    if np.isnan(avg):
                        avg_str = "N/A"
                    else:
                        avg_str = f"{avg:.4f}"
                    f.write(f"- {t.capitalize()}: F1 médio = {avg_str}, perguntas = {count}\n")

                if tree_stats_lines:
                    f.write("\n## Estrutura das árvores por paper\n\n")
                    for block in tree_stats_lines:
                        f.write(block + "\n\n")
        except Exception as e:
            logging.warning(f"Falha ao escrever arquivo de saída {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAPTOR+SOM+LLaMA on QASPER (subset)")
    parser.add_argument("--num-papers", type=int, default=100, help="Number of QASPER papers to evaluate")
    parser.add_argument("--num-questions-per-paper", type=int, default=5, help="Max questions per paper to evaluate")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Model name to use (Ollama or Gemini, depending on backend)")
    parser.add_argument("--split", type=str, default="validation", help="QASPER split to use (e.g., validation, test)")
    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama", "gemini"],
        help="LLM backend to use for summarization and QA",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="som",
        choices=["som", "dbscan", "gmm","full"],
        help="Clustering backend to use for RAPTOR tree (som, dbscan, gmm) ou 'full' para usar o documento inteiro como contexto",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to a file where metrics and tree stats will be written (markdown)",
    )

    args = parser.parse_args()

    evaluate_qasper_subset(
        num_papers=args.num_papers,
        num_questions_per_paper=args.num_questions_per_paper,
        model_name=args.model,
        split=args.split,
        backend=args.backend,
        cluster_backend=args.cluster,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
