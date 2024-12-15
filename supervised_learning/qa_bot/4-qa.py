import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np
import sentence_transformers
import faiss
import os


def find_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question using BERT.

    Args:
        question (str): The question to answer.
        reference (str): The reference document from which to find the answer.

    Returns:
        str or None: The answer string if found, otherwise None.
    """
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Tokenize the inputs
    inputs = tokenizer.encode_plus(question, reference, add_special_tokens=True, return_tensors='tf')


    question_tok = tokenizer.tokenize(question)
    reference_tok = tokenizer.tokenize(reference)

    manual_token = ['[CLS]'] + question_tok + ['[SEP]'] + reference_tok + ['[SEP]']

    input_ids = inputs['input_ids']  # Shape: (1, sequence_length), type tensor
    attention_mask = inputs['attention_mask']  # Shape: (1, sequence_length)
    token_type_ids = inputs['token_type_ids']  # Shape: (1, sequence_length)

    # Load the BERT QA model from TensorFlow Hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Perform inference
    outputs = model([
        input_ids,
        attention_mask,
        token_type_ids
    ])

    # Find the start and end indices with the highest logits and convert to int
    start_index = int(tf.argmax(outputs[0][0][1:])) + 1
    end_index = int(tf.argmax(outputs[1][0][1:])) + 1

    # Validate the indices
    if end_index <= start_index:
        return None

    # Convert token IDs to tokens with manually made tokens
    answer = manual_token[start_index: end_index + 1]

    # Combine tokens into a single string
    answer = tokenizer.convert_tokens_to_string(answer)

    return answer


def answer_loop(corpus_path, reference=None):
    """answers questions based on given reference"""
    while True:
        question = input("Q: ")

        question_lower = question.lower()

        if question_lower in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break
        if corpus_path is not None:
            reference = semantic_search(corpus_path, question)
            answer = find_answer(question_lower, reference)
        if answer is None:
            print("Sorry, I do not understand your question.")
        else:
            print("A: ", answer)

def question_answer(corpus_path):
    """takes in corpus_path, creates loop"""
    answer_loop(corpus_path)


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
        with open(os.path.join(corpus_path, list_docs[idx]), 'r', encoding='utf-8', errors='ignore') as file:
            # print(os.path.join(corpus_path, list_docs[idx]))
            text = file.read()
            break
    return text