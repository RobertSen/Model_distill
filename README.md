# 虚假夸大模型

## 运行手册
### 代码结构
```
root
|-- data: 数据目录
|    |-- config.ini: 模型训练、预测时所需参数
|-- model: 模型目录，训练得到的模型也将存储在该目录
|    |-- class_id.txt: 虚假夸大的细分类别及其对应id
|    |-- vocab.txt: ernie、gru、textcnn公用的字典
|-- pyweb: 部署网络服务的代码
|    |-- bin:
|    |    |-- run.sh: 启动网络服务的脚本
|    |-- src:
|    |    |-- fake_or_exaggeration_service.py: 网络服务的实现代码，基于tornado
|    |-- templates:
|    |    |-- index.html: 网页模板
|-- src:
|    |-- nets: 神经网络结构
|    |    |-- __init__.py:
|    |    |-- ernie_for_sequence_classification.py: 定制化的ernie，修改了其forward的参数和返回格式以使其适配训练的脚本
|    |    |-- gru.py: gru模型
|    |    |-- textcnn.py: textcnn模型
|    |-- dygraph.py: 实现模型训练、蒸馏等函数功能
|    |-- ernie_tokenizer.py: 将句子处理为模型接受的id序列
|    |-- label_encoder.py: 类别名称和id转换
|    |-- load_data.py: 加载训练、final_eval、unmark数据
|    |-- pygtrie.py: ernie_tokenizer依赖
|    |-- run_distill.py: 执行模型训练
|    |-- static_predict.py: 加载静态图模型进行预测
|-- README.md:
```


### 运行准备

确认配置参数文件"data/config.ini"，修改该文件或新建自己的配置文件。

### 训练模型

python2运行

执行目录： root

执行命令： python src/run_distill.py config_path [uniqid]

参数：
1. config_path: 自己修改后的配置参数文件
2. uniqid：可选参数。模型训练输出的数据文件名均会带上uniqid以区分。根据实际的uniqid参数，模型有三种处理方式：
    1. uniqid未给定，则模型uniqid为default
    2. 给定uniqid：若uniqid = ${time}，则以当前时间（精确到天）为uniqid，格式："%Y%M%D"；否则，以uniqid指定的参数值为uniqid

### 模型预测

执行目录： root

执行命令： echo "xxx" | python src/static_predict.py config_path [uniqid]

参数：
1. config_path: 自己修改后的配置参数文件
2. uniqid：可选参数。预测时所用的模型有其uniqid，通过uniqid指定用的模型。根据实际的uniqid参数，模型有三种处理方式：
    1. uniqid未给定，则模型uniqid为default
    2. 给定uniqid：若uniqid = ${time}，则以当前时间（精确到天）为uniqid，格式："%Y%M%D"；否则，以uniqid指定的参数值为uniqid

### 训练样本标签纠错

执行目录： root

步骤1:
    执行命令： python src/wrong_label_detect.py config_path [uniqid]

    参数：
    1. config_path: 自己修改后的配置参数文件
    2. uniqid：可选参数。预测时所用的模型有其uniqid，通过uniqid指定用的模型。根据实际的uniqid参数，模型有三种处理方式：
        1. uniqid未给定，则模型uniqid为default
        2. 给定uniqid：若uniqid = ${time}，则以当前时间（精确到天）为uniqid，格式："%Y%M%D"；否则，以uniqid指定的参数值为uniqid

步骤2:
    执行命令： python src/wrong_label_clean.py config_path

    参数：
    1. config_path: 自己修改后的配置参数文件


## 模型训练方案
ernie模型finetune后作为teacher模型，蒸馏得到textcnn模型。

### 蒸馏结构
![蒸馏结构][distilling_structure]

## 实验
### 数据集
|数据类型|数据量|
|:---:|:---:|
|训练数据|11190|
|final eval数据|1000|
|unmark数据|500000|

### 实验结果
由表可知：
1. 蒸馏得到的textcnn、gru模型效果均明显高于独立训练的textcnn、gru。
2. 同训练情况下，textcnn模型效果优于gru
3. 蒸馏时加入大量未标注数据能显著提升蒸馏效果
![模型蒸馏效果][distilling_performance]


参考论文：[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)

[distilling_performance]: http://agroup.baidu.com/api/static/bj/-5c768808234704894cc3128c15f001f80371bc2b?filename=%E6%A8%A1%E5%9E%8B%E8%92%B8%E9%A6%8F%E6%95%88%E6%9E%9C.png
[distilling_structure]: http://agroup.baidu.com/api/static/bj/-477a8a53bcb4c1dadd6a7cb01670cfe80e38dda3?filename=%E6%A8%A1%E5%9E%8B%E8%92%B8%E9%A6%8F.png
