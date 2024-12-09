#!/usr/bin/env python3
"""creates tf class Dataset"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """class dataset"""

    def __init__(self, batch_size, max_len):
        """init func"""
        self.max_length = max_len
        self.batch_size = batch_size
        exam, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
        self.data_train, self.data_valid = exam['train'], exam['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        self.data_train = self.data_train.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len))

        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(buffer_size=20000)

        self.data_train = self.data_train.padded_batch(
            batch_size,
            # this is apparently what we do if you have a variable length
            padded_shapes=([None], [None])
        )

        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len))

        self.data_valid = self.data_valid.padded_batch(
            batch_size,
            padded_shapes=([None], [None])
        )

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

    def encode(self, pt, en):
        """encodes"""
        pt_encode = self.tokenizer_pt.encode(pt.numpy().decode('utf-8'))
        en_encode = self.tokenizer_en.encode(en.numpy().decode('utf-8'))

        return pt_encode, en_encode

    def tf_encode(self, pt, en):
        """is wrapper for tf encode method"""

        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded

    def checks_length(self, pt, en):
        """filtering function for data_train"""
        if len(pt) > self.max_length or len(en) > self.max_length:
            return None
        return pt, en

    def tf_checks_length(self, pt, en):
        """tf wrapper for filtering function"""

        pt_filtered, en_filtered = tf.py_function(
            func=self.checks_length,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        return pt_filtered, en_filtered
