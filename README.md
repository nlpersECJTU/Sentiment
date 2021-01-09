# Sentiment Analysis

- Xu 等, 2020 - Aspect Sentiment Classification with Aspect-Specific Opinion Spans(EMNLP2020)<a href="https://www.aclweb.org/anthology/2020.emnlp-main.288/"> paper</a><a href="https://github.com/xuuuluuu/Aspect-Sentiment-Classification"> code</a><br/>
论文基于结构化attention模型，提出了多个CRF用来建模aspect的观点跨度信息。具体来说，加入一个latent label z属于{Yes,No}表示每个词是否在 aspect跨度之内，
再用多个CRF提取aspect的观点跨度信息。

- xue和li等，2018-Aspect Based Sentiment Analysis with Gated Convolutional Networks(acl 2018)<a href="https://www.aclweb.org/anthology/P18-1234/"> paper</a><a href=" https://github.com/wxue004cs/GCAE"> code</a><br/>
论文采用卷积神经网络和门控制机制比之前的lstm+注意力机制模型简洁、运算效率高，门控机制的作用可以在给定的aspect信息中，有选择的提取aspect情感信息

- Effective LSTMs for Target-Dependent Sentiment Classification(coling2015) <a href="https://arxiv.org/pdf/1512.01100v2.pdf"> paper</a><a href="http://ir.hit.edu.cn/~dytang/ "> code</a><br/>
论文基于LSTM神经网络构建模型，可以有效的识别基于Aspect实体情感分析这个子任务，文中通过标记，提取文本中实体信息来推断出实体的情感极性

DomBERT: Domain-oriented Language Model for Aspect-based Sentiment Analysis(EMNLP2020)<a href="https://www.aclweb.org/anthology/2020.findings-emnlp.156/"> paper</a><a href="https://github.com/howardhsu/BERT-for-RRC-ABSA"> code</a><br/>
论文旨在学习一个面向邻域的语言模型。通过采样领域内样本和相近领域的样本，保留BERT的MLM任务，去掉NSP任务，加入领域分类任务以学习到充分的领域知识。

Convolution over Hierarchical Syntactic and Lexical Graphs for Aspect Level Sentiment Analysis(EMNLP2020)<a href="https://www.aclweb.org/anthology/2020.emnlp-main.286/"> paper</a><a href="https://github.com/NLPWM-WHU/BiGCN"> code</a><br/>
论文考虑到了语料级别的词共现信息，认为不同的词和词之间的句法关系应该要区别对待。具体来说，论文在建立句法树时为不同句法关系的两个词分配不同的边；统计语料中的词共现频率作为词法树的辺，最后利用多层GCN融合句法树，词法树，词的上下文表示三种信息后进行情感分类。
