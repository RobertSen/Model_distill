#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
"""

import sys
import os
import logging

def check_dir(dir_address):
    """检测目录是否存在
        1. 若不存在则新建
        2. 若存在但不是文件夹，则报错
        3. 若存在且是文件夹则返回
    """
    if not os.path.isdir(dir_address):
        if os.path.exists(dir_address):
            raise ValueError("specified address is not a directory: %s" % dir_address)
        else:
            logging.info("create directory: %s" % dir_address)
            os.makedirs(dir_address)


if __name__ == "__main__":
    pass
