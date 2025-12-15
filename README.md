# Avaliação de Técnicas de Clusterização em Arquiteturas RAG com RAPTOR Modificado

Este repositório contém uma versão modificada do código do artigo RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (https://github.com/parthsarthi03/raptor), com experimentos focados na avaliação do impacto de diferentes algoritmos de clusterização na construção da árvore hierárquica de chunks para sistemas RAG (Retrieval-Augmented Generation).

# Objetivo:
Investigar como diferentes métodos de agrupamento (GMM, DBSCAN, SOM) afetam a formação dos chunks textuais e o desempenho do sistema RAG em tarefas de QA em documentos longos (usando o dataset QASPER).

O script principal qasper_eval_som_full.py realiza:

- Construção da árvore de sumarização inter-documento com base en diferentes estratégias de clusterização (GMM, DBSCAN, SOM);

- Geração de respostas a perguntas com modelos de linguagem (LLaMA3.2, Gemini);

- Cálculo da métrica F1 de sobreposição de tokens entre a resposta gerada e as respostas ouro (gold);

# Resultados:

F1 médio: 23.25%

Algoritmo com melhor desempenho: SOM com distância cosseno


# Referências:

Sarthi, Parth et al. “RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.” ArXiv abs/2401.18059 (2024): n. pag.

QASPER Dataset: https://allenai.org/data/qasper

Sentence-BERT: https://www.sbert.net/
