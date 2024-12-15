#!/usr/bin/env python3
"""performs semantic search on a corpus of documents"""
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import sentence_transformers
import faiss
import os


def semantic_search(corpus_path, sentence):
    """does semantic search"""
    model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')


    list_docs = []

    if not os.path.isdir(corpus_path):
        raise ValueError("corpus path does not exist")

    for doc in os.listdir(corpus_path):
        # if not os.path.isfile(doc):
        #     continue
        if doc.endswith('.md'):
            list_docs.append(doc)
            print(f"{doc} has been added to list")

    corpus_embeddings = model.encode(list_docs, convert_to_numpy=True, show_progress_bar=True)

    d = corpus_embeddings.shape[1]

    index = faiss.IndexFlatL2(d)

    index.add(corpus_embeddings)

    query = sentence
    query_embedding = model.encode([query], convert_to_numpy=True)

    distances, indices = index.search(query_embedding, 1)

    for i, idx in enumerate(indices[0]):
        # print(f"Rank {i + 1}: {list_docs[idx]} (Distance: {distances[0][i]})")
        with open(os.path.join(corpus_path, list_docs[idx]), 'r') as file:
            # print(os.path.join(corpus_path, list_docs[idx]))
            text = file.read()
            break
    return text
