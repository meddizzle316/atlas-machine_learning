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


    def tokenize_dataset(self, data):
        """tokenizes the data"""
        pt_list = []
        en_list = []
        for pt, en in data:
            pt_list.append(pt.numpy().decode('utf-8'))
            en_list.append(en.numpy().decode('utf-8'))

        self.tokenizer_en.add_tokens(en_list)
        self.tokenizer_en.model_max_length = 2 ** 13

        self.tokenizer_pt.add_tokens(pt_list)
        self.tokenizer_pt.model_max_length = 2 ** 13

        return self.tokenizer_pt, self.tokenizer_en
