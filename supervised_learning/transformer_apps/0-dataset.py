#!/usr/bin/env python3
"""creates tf class Dataset"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """class dataset"""

    def __init__(self):
        """init func"""
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
        self.data_train, = examples['train']
        self.data_valid = examples['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """tokenizes dataset"""
        pt_list = []
        en_list = []
        for pt, en in data:
            pt_list.append(pt.numpy().decode("utf-8"))
            en_list.append(en.numpy().decode('utf-8'))

        def pt_iterator():
            for text in pt_list:
                yield text

        def en_iterator():
            for text in en_list:
                yield text

        english_base = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased')
        portuguese_base = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')

        # transformers.PreTrainedTokenizerFast.train_new_from_iterator()
        vocab_size = 2 ** 13

        new_english_tokenizer = english_base.train_new_from_iterator(
            text_iterator=en_iterator(),
            vocab_size=vocab_size,
        )
        new_portuguese_tokenizer = portuguese_base.train_new_from_iterator(
            text_iterator=pt_iterator(),
            vocab_size=vocab_size,
        )

        return new_portuguese_tokenizer, new_english_tokenizer
