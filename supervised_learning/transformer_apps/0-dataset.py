#!/usr/bin/env python3
"""creates tf class Dataset"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """class dataset"""

    def __init__(self):
        """init func"""
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
        self.data_train, self.data_valid = examples['train'], examples['validation']
        self.tokenizer_en = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.tokenizer_pt = transformers.BertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased')

    class Dataset():
        """loads and preps a dataset for machine translation"""

        def __init__(self):
            """
            creates the instance attributes:
                data_train - contains the ted_hrlr_translate/pt_to_en tf.data.Dataset train split, loaded as_supervided
                data_valid - contains the ted_hrlr_translate/pt_to_en tf.data.Dataset validate split, loaded as_supervided
                tokenizer_pt - Portuguese tokenizer created from the training set
                tokenizer_en - English tokenizer created from the training set
            """
            self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
            self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
            self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        def tokenize_dataset(self, data):
            """
            creates sub-word tokenizers for dataset

            data is a tf.data.Dataset whose examples are formatted as a tuple (pt, en)
                pt is the tf.Tensor containing the Portuguese sentence
                en is the tf.Tensor containing the corresponding English sentence

            The maximum vocab size should be set to 2**15

            Returns: tokenizer_pt, tokenizer_en
                tokenizer_pt is the Portuguese tokenizer
                tokenizer_en is the English tokenizer
            """
            f = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
            en_tok = f((en.numpy() for _, en in data.take(10)),
                       target_vocab_size=2 ** 15)
            pt_tok = f((pt.numpy() for pt, _ in data.take(10)),
                       target_vocab_size=2 ** 15)
            return pt_tok, en_tok
