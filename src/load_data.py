#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
"""

import logging
import os
import sys
import time
from common.data_io import get_attr_values

from sklearn.model_selection import train_test_split


class DataLoader(object):
    """ 数据加载类
    """
    def __init__(self, tokenizer, label_encoder):
        self.tokenizer = tokenizer
        self.id_2_token = {v: k for k, v in self.tokenizer.vocab.items()}
        self.label_encoder = label_encoder

    def gen_data(self,
            train_data_dir,
            unmark_data_dir=None,
            final_eval_data_dir=None,
            encoding="utf-8",
            test_ratio=0.2,
            random_state=None,
            is_shuffle=True,
            example_num=5):
        """生成训练数据
        [IN]  train_data_dir: string, 训练数据目录
              unmark_data_dir: string or None,  None则无unmark
              final_eval_data_dir: string or None, Nonez怎无
              encoding: string, file encoding
              test_ratio: float, 划分训练集和验证集的比例
              random_state: int, shuffle时的随机种子
              is_shuffle: boolean: boolean， 是否打乱顺序
              example_num: int, 示例展示数量
        """

        self.train_text_list, train_text_ids, self.train_label_list = \
                DataLoader.load_data(train_data_dir, self.tokenizer, encoding, True)

        train_label_ids = [[self.label_encoder.transform(x)] for x in self.train_label_list]

        if unmark_data_dir is not None:
            self.unmark_text_list, unmark_text_ids = \
                    DataLoader.load_data(unmark_data_dir, self.tokenizer, encoding, False)
        else:
            self.unmark_text_list, unmark_text_ids = list(), list()

        self.final_eval_text_list, final_eval_text_ids, self.final_eval_label_list = \
                DataLoader.load_data(final_eval_data_dir, self.tokenizer, encoding, True)

        # 划分训练集 测试集
        self.train_text, self.test_text, train_x, test_x, train_y, test_y = train_test_split(
                self.train_text_list, train_text_ids, train_label_ids, test_size=test_ratio,
                random_state=random_state, shuffle=is_shuffle)

        self.train_data = zip(train_x, train_y)
        self.eval_data = zip(test_x, test_y)

        empty_label_ids = [[-1] for i in range(len(final_eval_text_ids))]
        self.final_eval_data = zip(final_eval_text_ids, empty_label_ids)

        self.unmark_data = zip(unmark_text_ids)

        logging.info("label num = {}".format(self.label_encoder.size()))
        logging.info("train data num = {}".format(len(train_y)))
        logging.info("eval data num = {}".format(len(test_y)))
        logging.info("unmark data num = {}".format(len(unmark_text_ids)))
        logging.info("final eval data num = {}".format(len(empty_label_ids)))

        logging.info(u"数据样例：")
        for i in range(example_num):
            logging.info("\t".join([self.train_label_list[i], self.train_text_list[i]]))

        train_example = map(lambda ids: "/ ".join([self.id_2_token[x] for x in ids]), train_x[:example_num])
        example_list = zip(self.train_text[:example_num], train_example)
        for index, (cur_text, cur_str) in enumerate(example_list):
            logging.info("example #%d" % index)
            logging.info("text: %s" % cur_text)
            logging.info("train_str: %s" % cur_str)

    @staticmethod
    def load_data(data_dir, tokenizer, encoding="utf-8", has_label=False):
        """加载数据
        [IN] data_dir: string, 数据目录
             tokenizer: 切分工具
             encoding：string, 数据编码
             has_label: true则数据中包含标签
        [OUT] res: list[]
        """
        fetch_attrs = ["text"]
        if has_label:
            fetch_attrs.append("label")

        attr_values = get_attr_values(
            data_dir,
            fetch_list=fetch_attrs,
            encoding=encoding,
            )

        logging.info(u"数据样例：")
        for i in range(5):
            logging.info("\t".join([attr[i] for attr in attr_values]))

        if has_label:
            text_list, label_list = attr_values
        else:
            text_list = attr_values[0]

        logging.info("tokenizer encode start")
        start_time = time.time()
        text_ids = [tokenizer.encode(x)[0] for x in text_list]
        logging.info("cost time: %.4fs" % (time.time() - start_time))

        res = [text_list, text_ids]
        if has_label:
            res.append(label_list)

        return res

if __name__ == "__main__":
    from ernie_tokenizer import ErnieTokenizer
    from label_encoder import LabelEncoder

    DATA_DIR = "data/zui_mark_0826.txt"
    UNMARK_DATA_DIR = "data/unmark_data/zui_detected_shishi_0805.txt"
    FINAL_EVAL_DATA_DIR="data/eval_data/zui_first_evaluate.txt"

    LABEL_ENCODER_PATH = "model/class_id.txt"
    label_encoder = LabelEncoder(label_id_info=LABEL_ENCODER_PATH, isFile=True)

    VOCAB_PATH = "model/vocab.txt"
    tokenizer = ErnieTokenizer.from_pretrained(VOCAB_PATH)

    loader = DataLoader(tokenizer, label_encoder)
    loader.gen_data(
            train_data_dir=DATA_DIR,
            unmark_data_dir=UNMARK_DATA_DIR,
            final_eval_data_dir=FINAL_EVAL_DATA_DIR,
            encoding="utf-8",
            test_ratio=0.2,
            random_state=None,
            is_shuffle=True,
            example_num=5)

