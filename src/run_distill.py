#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
"""

import codecs
import configparser
import datetime
import logging
import os
import sys
import time

import numpy as np
import paddle.fluid as F
import paddle.fluid.dygraph as D

from dygraph import batch_infer
from dygraph import distill
from dygraph import infer
from dygraph import train
from ernie.modeling_ernie import ErnieModelForSequenceClassification
from ernie_tokenizer import ErnieTokenizer
from label_encoder import LabelEncoder
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % (_cur_dir))
from common.data_io import get_attr_values
from common.data_io import load_model
from common.logger import init_log
from load_data import DataLoader
from nets.ernie_for_sequence_classification import ErnieModelCustomized
from nets.gru import GRU
from nets.textcnn import TextCNN
from sklearn.metrics import classification_report
from utils import check_dir
init_log()



class DistillRunner(object):
    """知识蒸馏执行类
    """
    def __init__(self, config_path, uniqid=None):
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(config_path)
        if uniqid is not None:
            if uniqid == "${time}":
                now_time = datetime.datetime.now()
                uniqid = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d")
            # 添加当前uniqid
            self.config.set("DEFAULT", "uniqid", uniqid)
        logging.info("uniqid: {}".format(self.config.get("DEFAULT", "uniqid")))
        logging.info(self.config["DEFAULT"]["output_dir"])
        check_dir(self.config["DEFAULT"]["output_dir"])
        check_dir(self.config["DEFAULT"]["data_dir"])
        check_dir(self.config["DEFAULT"]["model_dir"])

        self.run_config = self.config["RUN"]
        self.data_config = self.config["DATA"]
        self.model_config = self.config["MODEL_PATH"]

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.run_config["cuda_visible_devices"])
        logging.info("gpu id : {}".format(self.run_config["cuda_visible_devices"]))

        self.label_encoder = LabelEncoder(label_id_info=self.model_config["label_encoder"], isFile=True)

        for label_id, label_name in sorted(self.label_encoder.id_label_dict.items(), key=lambda x:x[0]):
            logging.info("%d: %s" % (label_id, label_name))

        self.tokenizer = ErnieTokenizer.from_pretrained(self.model_config["tokenizer"])
        self.id_2_token = {v: k for k, v in self.tokenizer.vocab.items()}

    def run(self):
        """执行入口
        """
        self.data_loader = DataLoader(self.tokenizer, self.label_encoder)
        self.data_loader.gen_data(
                train_data_dir=self.data_config["train_data_path"],
                unmark_data_dir=self.data_config.get("unmark_data_path", None),
                final_eval_data_dir=self.data_config["eval_data_path"],
                encoding=self.data_config["encoding"],
                test_ratio=self.data_config.getfloat("test_ratio"),
                random_state=self.data_config.getint("random_state", None),
                is_shuffle=self.data_config.getboolean("is_shuffle"),
                example_num=self.data_config.getint("example_num"),
                )

        if self.run_config.getboolean("ernie_finetune"):
            best_acc, final_eval_acc = self.ernie_finetune()
            logging.info("[IMPORTANT]ernie performance: " \
                    "best acc = {:.4f}, final acc = {:.4f}".format(best_acc, final_eval_acc))

        if self.run_config.getboolean("train_textcnn"):
            best_acc, final_eval_acc = self.train_textcnn()
            logging.info("[IMPORTANT]textcnn performance: " \
                    "best acc = {:.4f}, final acc = {:.4f}".format(best_acc, final_eval_acc))

        if self.run_config.getboolean("distill_textcnn"):
            best_acc, final_eval_acc =  self.distill_textcnn(use_unmark_data=False)
            logging.info("[IMPORTANT]distill textcnn without unmark data performance: " \
                    "best acc = {:.4f}, final acc = {:.4f}".format(best_acc, final_eval_acc))

        if self.run_config.getboolean("distill_textcnn_with_unmark"):
            best_acc, final_eval_acc = self.distill_textcnn(use_unmark_data=True)
            logging.info("[IMPORTANT]distill textcnn with unmark data performance: " \
                    "best acc = {:.4f}, final acc = {:.4f}".format(best_acc, final_eval_acc))

        if self.run_config.getboolean("textcnn_to_static"):
            self.textcnn_to_static()
            logging.info("[IMPORTANT]textcnn model to static")

        if self.run_config.getboolean("train_gru"):
            best_acc, final_eval_acc = self.train_gru()
            logging.info("[IMPORTANT]gru performance: best acc = {:.4f}, final acc = {:.4f}"\
                    .format(best_acc, final_eval_acc))

        if self.run_config.getboolean("distill_gru"):
            best_acc, final_eval_acc =  self.distill_gru(use_unmark_data=False)
            logging.info("[IMPORTANT]distill gru without unmark data performance: " \
                    "best acc = {:.4f}, final acc = {:.4f}".format(best_acc, final_eval_acc))

        if self.run_config.getboolean("distill_gru_with_unmark"):
            best_acc, final_eval_acc = self.distill_gru(use_unmark_data=True)
            logging.info("[IMPORTANT]distill gru with unmark data performance: " \
                    "best acc = {:.4f}, final acc = {:.4f}".format(best_acc, final_eval_acc))


    def textcnn_to_static(self):
        """textcnn模型转静态图模型文件
        """
        textcnn_config = self.config["TEXTCNN"]
        distill_textcnn_config = self.config["DISTILL_TEXTCNN"]

        with D.guard():

            text_cnn = TextCNN(
                    num_class=self.label_encoder.size(),
                    vocab_size=textcnn_config.getint("vocab_size"),
                    emb_dim=textcnn_config.getint("emb_dim"),
                    num_filters=textcnn_config.getint("num_filters"),
                    fc_hid_dim=textcnn_config.getint("fc_hid_dim"),
                    use_cudnn=textcnn_config.getboolean("use_cudnn"),
                    )
            load_model(text_cnn, self.model_config["distill_textcnn_model_best"])
            fake_input = np.random.random([
                    distill_textcnn_config.getint("batch_size"),
                    distill_textcnn_config.getint("max_seq_len")]).astype('int64')
            fake_input_tensor = D.to_variable(fake_input)
            _, static_layer = D.TracedLayer.trace(text_cnn, inputs=[fake_input_tensor, None, True])

            # 将静态图模型保存为预测模型
            static_layer.save_inference_model(dirname=self.model_config["static_textcnn_model"])
            logging.info("save static textcnn to {}".format(self.model_config["static_textcnn_model"]))

    def train_gru(self):
        """人工标定数据训练gru模型
        """
        gru_config = self.config["GRU"]
        with D.guard():
            gru = GRU(
                    num_class=self.label_encoder.size(),
                    vocab_size=gru_config.getint("vocab_size"),
                    emb_dim=gru_config.getint("emb_dim"),
                    gru_dim=gru_config.getint("gru_dim"),
                    fc_hid_dim=gru_config.getint("fc_hid_dim"),
                    bi_direction=gru_config.getboolean("bi_direction"),
                    )
            load_model(gru, self.model_config["gru_model_best"])
            optimizer = F.optimizer.Adam(
                    learning_rate=gru_config.getfloat("learning_rate"),
                    parameter_list=gru.parameters())

            best_acc = train(gru, optimizer, \
                    self.data_loader.train_data, self.data_loader.eval_data, self.label_encoder,
                    model_save_path=self.model_config["gru_model"],
                    best_model_save_path=self.model_config["gru_model_best"],
                    epochs=gru_config.getint("epoch"),
                    batch_size=gru_config.getint("batch_size"),
                    max_seq_len=gru_config.getint("max_seq_len"),
                    print_step=gru_config.getint("print_step"),
                    )
            logging.info("gru best score: {}".format(best_acc))

        final_eval_acc = self.final_eval(
                gru,
                gru_config["final_eval_res"],
                gru_config["final_eval_diff"])
        return best_acc, final_eval_acc

    def distill_gru(self, use_unmark_data=True):
        """ernie作为teacher模型，知识蒸馏训练student模型gru
        [IN]  use_unmark_data: boolean, true则使用未标注数据
        [OUT] best_acc: float, gru训练时在验证集上的最佳acc
              final_eval_acc: float, gru在final_eval数据集上的最佳acc
        """
        gru_config = self.config["GRU"]
        distill_gru_config = self.config["DISTILL_GRU"]

        with D.guard():

            ernie = ErnieModelCustomized.from_pretrained(
                    self.model_config["ernie_pretrain"],
                    num_labels=self.label_encoder.size())
            load_model(ernie, self.model_config["ernie_model_best"])

            gru = GRU(
                    num_class=self.label_encoder.size(),
                    vocab_size=gru_config.getint("vocab_size"),
                    emb_dim=gru_config.getint("emb_dim"),
                    gru_dim=gru_config.getint("gru_dim"),
                    fc_hid_dim=gru_config.getint("fc_hid_dim"),
                    bi_direction=gru_config.getboolean("bi_direction"),
                    )

            load_model(gru, self.model_config["distill_gru_model_best"])
            optimizer = F.optimizer.Adam(
                    learning_rate=distill_gru_config.getfloat("learning_rate"),
                    parameter_list=gru.parameters())

            best_acc = distill(ernie, gru, optimizer, \
                    self.data_loader.train_data, self.data_loader.eval_data, self.label_encoder,
                    unmark_data=self.data_loader.unmark_data if use_unmark_data else None,
                    model_save_path=self.model_config["distill_gru_model"],
                    best_model_save_path=self.model_config["distill_gru_model_best"],
                    epochs=distill_gru_config.getint("epoch"),
                    batch_size=distill_gru_config.getint("batch_size"),
                    max_seq_len=distill_gru_config.getint("max_seq_len"),
                    print_step=distill_gru_config.getint("print_step"),
                    )

            logging.info("distill gru best score: {}".format(best_acc))

        final_eval_acc = self.final_eval(
                gru,
                distill_gru_config["final_eval_res"],
                distill_gru_config["final_eval_diff"])
        return best_acc, final_eval_acc

    def train_textcnn(self):
        """人工标定数据训练textcnn模型
        """
        textcnn_config = self.config["TEXTCNN"]
        with D.guard():
            text_cnn = TextCNN(
                    num_class=self.label_encoder.size(),
                    vocab_size=textcnn_config.getint("vocab_size"),
                    emb_dim=textcnn_config.getint("emb_dim"),
                    num_filters=textcnn_config.getint("num_filters"),
                    fc_hid_dim=textcnn_config.getint("fc_hid_dim"),
                    use_cudnn=textcnn_config.getboolean("use_cudnn"),
                    )

            load_model(text_cnn, self.model_config["textcnn_model_best"])
            optimizer = F.optimizer.Adam(
                    learning_rate=textcnn_config.getfloat("learning_rate"),
                    parameter_list=text_cnn.parameters())

            best_acc = train(text_cnn, optimizer, \
                    self.data_loader.train_data, self.data_loader.eval_data, self.label_encoder,
                    best_acc = 0,
                    model_save_path=self.model_config["textcnn_model"],
                    best_model_save_path=self.model_config["textcnn_model_best"],
                    epochs=textcnn_config.getint("epoch"),
                    batch_size=textcnn_config.getint("batch_size"),
                    max_seq_len=textcnn_config.getint("max_seq_len"),
                    print_step=textcnn_config.getint("print_step"),
                    )
            logging.info("textcnn best score: {}".format(best_acc))

        final_eval_acc = self.final_eval(
                text_cnn,
                textcnn_config["final_eval_res"],
                textcnn_config["final_eval_diff"])
        return best_acc, final_eval_acc

    def distill_textcnn(self, use_unmark_data=True):
        """ernie作为teacher模型，知识蒸馏训练student模型textcnn
        [IN]  use_unmark_data: boolean, true则使用未标注数据
        [OUT] best_acc: float, textcnn训练时在验证集上的最佳acc
              final_eval_acc: float, textcnn在final_eval数据集上的最佳acc
        """
        textcnn_config = self.config["TEXTCNN"]
        distill_textcnn_config = self.config["DISTILL_TEXTCNN"]

        with D.guard():

            ernie = ErnieModelCustomized.from_pretrained(
                    self.model_config["ernie_pretrain"],
                    num_labels=self.label_encoder.size())
            load_model(ernie, self.model_config["ernie_model_best"])

            text_cnn = TextCNN(
                    num_class=self.label_encoder.size(),
                    vocab_size=textcnn_config.getint("vocab_size"),
                    emb_dim=textcnn_config.getint("emb_dim"),
                    num_filters=textcnn_config.getint("num_filters"),
                    fc_hid_dim=textcnn_config.getint("fc_hid_dim"),
                    use_cudnn=textcnn_config.getboolean("use_cudnn"),
                    )

            load_model(text_cnn, self.model_config["distill_textcnn_model_best"])
            optimizer = F.optimizer.Adam(
                    learning_rate=distill_textcnn_config.getfloat("learning_rate"),
                    parameter_list=text_cnn.parameters())

            best_acc = distill(ernie, text_cnn, optimizer, \
                    self.data_loader.train_data, self.data_loader.eval_data, self.label_encoder,
                    unmark_data=self.data_loader.unmark_data if use_unmark_data else None,
                    model_save_path=self.model_config["distill_textcnn_model"],
                    best_model_save_path=self.model_config["distill_textcnn_model_best"],
                    epochs=distill_textcnn_config.getint("epoch"),
                    batch_size=distill_textcnn_config.getint("batch_size"),
                    max_seq_len=distill_textcnn_config.getint("max_seq_len"),
                    print_step=distill_textcnn_config.getint("print_step"),
                    )

            logging.info("distill textcnn best score: {}".format(best_acc))

        final_eval_acc = self.final_eval(
                text_cnn,
                distill_textcnn_config["final_eval_res"],
                distill_textcnn_config["final_eval_diff"])
        return best_acc, final_eval_acc

    def ernie_finetune(self):
        """人工标定数据训练ernie模型
        """
        ernie_config = self.config["ERNIE"]
        with D.guard():
            ernie = ErnieModelCustomized.from_pretrained(
                    self.model_config["ernie_pretrain"],
                    num_labels=self.label_encoder.size())
            load_model(ernie, self.model_config["ernie_model_best"])
            logging.info("ernie learning rate: {}".format(ernie_config.getfloat("learning_rate")))

            optimizer = F.optimizer.Adam(
                    ernie_config.getfloat("learning_rate"),
                    parameter_list=ernie.parameters())

            best_acc = train(ernie, optimizer, \
                    self.data_loader.train_data, self.data_loader.eval_data, self.label_encoder,
                    model_save_path=self.model_config["ernie_model"],
                    best_model_save_path=self.model_config["ernie_model_best"],
                    epochs=ernie_config.getint("epoch"),
                    batch_size=ernie_config.getint("batch_size"),
                    max_seq_len=ernie_config.getint("max_seq_len"),
                    print_step=ernie_config.getint("print_step"),
                    )
            logging.info("ernie best score: {}".format(best_acc))

        final_eval_acc = self.final_eval(
                ernie,
                ernie_config["final_eval_res"],
                ernie_config["final_eval_diff"])
        return best_acc, final_eval_acc

    def final_eval(self, model, res_path, diff_path, batch_size=32):
        """final_eval数据集评估函数
        """
        logging.info("final eval start")
        start_time = time.time()
        with D.guard():
            logits_list, _ = batch_infer(model, self.data_loader.final_eval_data, batch_size=batch_size)
        logging.info("cost time: %.4fs" % (time.time() - start_time))

        pred_list = np.argmax(logits_list, axis=-1)
        pred_name_list = map(lambda x: self.label_encoder.inverse_transform(x), pred_list)
        #pred_trans_list = pred_name_list
        pred_trans_list = [x if x == u"无风险" else u"有风险" for x in pred_name_list]
        logging.info("\n" + classification_report(
                self.data_loader.final_eval_label_list, pred_trans_list, digits=4))
        final_eval_acc = (np.array(self.data_loader.final_eval_label_list) == np.array(pred_trans_list))\
                .astype(np.float32).mean()
        logging.info("final eval acc = {:.4f}".format(final_eval_acc))

        logging.info("save eval res start")
        start_time = time.time()
        with codecs.open(res_path, "w", "utf-8") as wf, \
                codecs.open(diff_path, "w", "utf-8") as wf2:
            for pred_name, label_name, cur_text in \
                    zip(pred_trans_list, self.data_loader.final_eval_label_list, self.data_loader.final_eval_text_list):
                wf.write("\t".join([pred_name, label_name, cur_text]) + "\n")
                if pred_name != label_name:
                    wf2.write("\t".join([pred_name, label_name, cur_text]) + "\n")
        logging.info("cost time: %.4fs" % (time.time() - start_time))
        return final_eval_acc


if __name__ == "__main__":
    config_path = sys.argv[1]
    uniqid = sys.argv[2] if len(sys.argv) > 2 else None
    runner = DistillRunner(config_path, uniqid)
    runner.run()
