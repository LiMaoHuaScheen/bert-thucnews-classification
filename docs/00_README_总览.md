# 00_README_总览

这是文本线第 1 步「中文文本分类」的完整文档包。

| 文档 | 内容 |
|------|------|
| [01_方案说明.md](01_方案说明.md) | 实验背景、目标与方案设计 |
| [02_环境搭建.md](02_环境搭建.md) | 环境准备、依赖安装、模型与数据下载 |
| [03_操作步骤.md](03_操作步骤.md) | 从零到一的可复现操作步骤 |
| [04_实验报告.md](04_实验报告.md) | 完整实验数据与结果分析 |
| [06_代码说明.md](06_代码说明.md) | 代码结构与关键实现 |

## 一句话结果

在 THUCNews 10 分类上，BERT-base-chinese 微调达到 **96.65%**，从头训练 **87.86%**，TF-IDF+LinearSVC 基线 **88.13%**。预训练带来 **+8.8 个百分点** 的提升。

## 目录结构

~~~
bert-thucnews-classification/
├── README.md
├── requirements.txt
├── download_data.py
├── code/  (train_bert.py, plot_bert.py)
├── figures/  (learning_curve.png, accuracy_bar.png)
├── results/  (result_summary.json)
└── docs/  (本目录 6 个文档)
~~~