#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
"""

import sys
import codecs
import configparser
from collections import defaultdict
from collections import namedtuple


def wrong_label_clean(config):
    """针对检测出的错误样本 人工确认后 批量修改
    """

    label_clean_config = config["LABEL_CLEAN"]
    wrong_label_path = label_clean_config["wrong_label_res"]
    file_encoding = label_clean_config["encoding"]
    modify_suffix = label_clean_config["modify_suffix"]


    modify_record_dict = defaultdict(dict)
    with codecs.open(wrong_label_path, "r", file_encoding) as rf:
        for line in rf:
            parts = line.strip("\n").split("\t")
            modify_label = parts[0]
            text = parts[2]
            file_path = parts[3]
            file_index = parts[4]
            modify_record_dict[file_path][file_index] = (modify_label, text)

    for file_path, cur_modify_dict in modify_record_dict.items():
        with codecs.open(file_path, "r", file_encoding) as rf, \
                codecs.open(file_path + "." + modify_suffix, "w", file_encoding) as wf:
            for index, line in enumerate(rf):
                parts = line.rstrip("\n").split("\t")
                if index == 0:
                    Record = namedtuple("record", parts)

                record = Record(*parts)

                if str(index) in cur_modify_dict:
                    cur_modify_label, cur_text = cur_modify_dict[str(index)]
                    assert index != 0, "modify index == 0, please check"
                    assert cur_text == record.text, \
                            ("text not equal at line #%d:\nori_text: %s\nmodify_text: %s" % \
                            ((index + 1), record.text, cur_text)).encode("utf-8")

                    record = record._replace(label=cur_modify_label)

                record_str = "\t".join([getattr(record, attr_name) for attr_name in record._fields])
                wf.write(record_str + "\n")


if __name__ == "__main__":
    config_path = sys.argv[1]

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path)

    wrong_label_clean(config)
