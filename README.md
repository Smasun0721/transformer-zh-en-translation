# Transformer 中英翻译

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Model-Transformer-1E90FF)
![Task](https://img.shields.io/badge/Task-Machine%20Translation-6A5ACD)

从零实现 Transformer 并完成中译英训练、验证和 beam search 推理的项目，适合学习注意力机制、位置编码和序列到序列建模。

## 目录

1. [项目背景](#项目背景)
2. [数据说明](#数据说明)
3. [模型结构](#模型结构)
4. [训练配置](#训练配置)
5. [实验结果](#实验结果)
6. [环境要求](#环境要求)
7. [安装](#安装)
8. [快速开始](#快速开始)
9. [项目结构](#项目结构)
10. [复现说明](#复现说明)
11. [后续优化](#后续优化)

## 项目背景

机器翻译是自然语言处理中的经典序列到序列任务。Transformer 通过自注意力机制替代循环结构，能够并行建模长距离依赖，是当前大模型的基础架构之一。

本项目从零实现 Transformer 的各个子模块，并在 10 万对中英语料上完成训练与推理，用于理解注意力机制、位置编码、编码器-解码器结构和训练技巧。

## 数据说明

| 文件 | 内容 | 规模 |
| --- | --- | ---: |
| src/data/chinese.txt | 中文训练语料 | 100,000 行，约 13.6 MB |
| src/data/english.txt | 英文训练语料 | 100,000 行，约 17.8 MB |

数据格式为每行一个句子，中英文文件按行对齐，中文已按空格分词。样例：

| 语言 | 样例 |
| --- | --- |
| 中文 | 1998年 , 经过 统一 部署 , 伊犁州 , 地 两 级 党委 开始 尝试 ... |
| 英文 | in 1998 , the yili autonomous prefecture cpc committee ... |

## 模型结构

![Transformer 架构](docs/architecture.png)

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| 输入嵌入 | src/dm01_input.py | 词嵌入 + 位置编码 |
| 注意力 | src/dm02_encoder_element.py | 缩放点积注意力、多头注意力、前馈网络、LayerNorm |
| 编码器 | src/dm03—dm05 | 编码器子层、编码器层和堆叠 |
| 解码器 | src/dm06—dm07 | 解码器子层、解码器层和堆叠 |
| 输出层 | src/dm08_output.py | Linear + Softmax 输出 |
| 模型组装 | src/dm09_transformer.py | 完整 Transformer |
| 训练/推理 | src/NiuTrans_train.py、src/inference.py | 训练、验证、beam search |

## 训练配置

| 参数 | 取值 |
| --- | --- |
| d_model | 256 |
| n_head | 8 |
| n_layer | 4 |
| d_ff | 1024 |
| dropout | 0.2 |
| batch_size | 64 |
| epochs | 150 |
| optimizer | Adam |
| learning_rate | 1e-3 |
| warmup_epochs | 2 |
| scheduler | CosineAnnealing + warmup |
| gradient_clip | 1.0 |

训练脚本支持断点续训（checkpoint）、混合精度、学习率调度和最优模型保存。

## 实验结果

原始训练日志约 212,456 个训练 step、150 个 epoch：

| 指标 | 数值 |
| --- | ---: |
| Val Loss | 2.3622 |
| Val Accuracy | 0.4802 |
| Val PPL | 10.61 |

![Transformer 训练曲线](reports/training_overview.png)

训练曲线显示 Loss 从约 9.4 降到约 2.7，学习率经过 warmup 后按余弦衰减，梯度范数保持在稳定区间。

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- 依赖见 requirements.txt

建议使用带 CUDA 的 GPU 训练完整 150 个 epoch；CPU 环境可用于模块测试和小规模调试。

## 安装

~~~bash
git clone https://github.com/Smasun0721/transformer-zh-en-translation.git
cd transformer-zh-en-translation
pip install -r requirements.txt
~~~

## 快速开始

训练：

~~~bash
cd src
python NiuTrans_train.py
~~~

推理：

~~~bash
cd src
python inference.py
~~~

训练会在 src/checkpoints 下生成 best_model.pt 和 last_checkpoint.pt。

## 项目结构

~~~text
transformer-zh-en-translation/
├── src/
│   ├── dm01_input.py
│   ├── dm02_encoder_element.py
│   ├── dm03_encoder_sublayer.py
│   ├── dm04_encoder_layer.py
│   ├── dm05_encoder.py
│   ├── dm06_decoder_layer.py
│   ├── dm07_decoder.py
│   ├── dm08_output.py
│   ├── dm09_transformer.py
│   ├── NiuTrans_train.py
│   ├── inference.py
│   ├── plot_metrics.py
│   └── data/
│       ├── chinese.txt
│       └── english.txt
├── reports/
│   ├── training_log.csv
│   └── training_overview.png
├── docs/architecture.png
├── requirements.txt
└── README.md
~~~

## 复现说明

- 训练日志 reports/training_log.csv 记录了每个 logging step 的 Loss、Acc、PPL、LR 和 GradNorm。
- 训练权重约 240MB，未直接提交到仓库；可先运行训练脚本生成，或放到 GitHub Releases / Hugging Face 后在 README 中补充下载地址。
- 推理脚本使用训练时同一套空格分词词表，避免推理阶段重建词表造成索引错位。

## 后续优化

- 使用 BLEU / chrF 等机器翻译指标评估生成质量。
- 增加 BPE/SentencePiece 子词分词和长度惩罚。
- 尝试梯度累积、多卡训练和模型导出（ONNX / TorchScript）。
- 提供 FastAPI 翻译服务或 Web Demo。
