import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np


def question_answer(question, reference):
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


def answer_loop(reference):
    """answers questions based on given reference"""
    while True:
        question = input("Q: ")

        question_lower = question.lower()

        if question_lower in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break
        answer = question_answer(question_lower, reference)
        if answer is None:
            print("Sorry, I do not understand your question.")
        else:
            print("A: ", answer)