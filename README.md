# Sentiment Analysis

- Xu 等, 2020 - Aspect Sentiment Classification with Aspect-Specific Opinion Spans(EMNLP2020)<a href="https://www.aclweb.org/anthology/2020.emnlp-main.288/"> paper</a><a href="https://github.com/xuuuluuu/Aspect-Sentiment-Classification"> code</a><br/>
论文基于结构化attention模型，提出了多个CRF用来建模aspect的观点跨度信息。具体来说，加入一个latent label z属于{Yes,No}表示每个词是否在 aspect跨度之内，
再用多个CRF提取aspect的观点跨度信息。

- xue和li等，2018-Aspect Based Sentiment Analysis with Gated Convolutional Networks(acl 2018)<a href="https://www.aclweb.org/anthology/P18-1234/"> paper</a><a href=" https://github.com/wxue004cs/GCAE"> code</a><br/>
论文采用卷积神经网络和门控制机制比之前的lstm+注意力机制模型简洁、运算效率高，门控机制的作用可以在给定的aspect信息中，有选择的提取aspect情感信息

- Effective LSTMs for Target-Dependent Sentiment Classification(coling2015) <a href="https://arxiv.org/pdf/1512.01100v2.pdf"> paper</a><a href="http://ir.hit.edu.cn/~dytang/ "> code</a><br/>
论文基于LSTM神经网络构建模型，可以有效的识别基于Aspect实体情感分析这个子任务，文中通过标记，提取文本中实体信息来推断出实体的情感极性

- DomBERT: Domain-oriented Language Model for Aspect-based Sentiment Analysis(EMNLP2020)<a href="https://www.aclweb.org/anthology/2020.findings-emnlp.156/"> paper</a><a href="https://github.com/howardhsu/BERT-for-RRC-ABSA"> code</a><br/>
论文旨在学习一个面向领域的语言模型。通过采样领域内样本和相近领域的样本，保留BERT的MLM任务，去掉NSP任务，加入领域分类任务以学习到充分的领域知识。

- Convolution over Hierarchical Syntactic and Lexical Graphs for Aspect Level Sentiment Analysis(EMNLP2020)<a href="https://www.aclweb.org/anthology/2020.emnlp-main.286/"> paper</a><a href="https://github.com/NLPWM-WHU/BiGCN"> code</a><br/>
论文考虑到了语料级别的词共现信息，认为不同的词和词之间的句法关系应该要区别对待。具体来说，论文在建立句法树时为不同句法关系的两个词分配不同的边；统计语料中的词共现频率作为词法树的辺，最后利用多层GCN融合句法树，词法树，词的上下文表示三种信息后进行情感分类。

- Transformation Networks for target-oriented sentiment classification(acl 2018)<a href="https://ai.tencent.com/ailab/media/publications/acl/Transformation_Networks_for_Target-Oriented_Sentiment_Classification.pdf"> paper</a><a href="https://github.com/lixin4ever/TNet"> code</a><br/>
论文作者针对之前面向目标的情感分类问题的主流方法Rnn+Attention方法中存在的缺点提出了TNet模型，计算target与句子中每个词权重，针对处理过程中的句子信息的损失作者还提出上下文保存机制，最后通过一层卷积层，可以更好的提取与target相关的短语的情感极性

- Disconnected Recurrent Neural Networks for Text Categorization(acl 2018)<a href="https://www.aclweb.org/anthology/P18-1215.pdf"> paper</a>
论文作者根据Rnn,Cnn在处理长文本中存在的缺点，提出了不连续、间断的Rnn，限制Rnn信息传输的步长，这样使得非连续的Rnn网络结合了传统的Rnn网络和Cnn网络的优点

- Attentional Encoder Network for Targeted Sentiment Classification（acl 2019）<a href="https://arxiv.org/pdf/1902.09314.pdf"> paper</a><a href="https://github.com/
songyouwei/ABSA-PyTorch/tree/aen."> code</a><br/>
与以往大多数RNN+Attention的思路不同，作者在这篇论文里给出了注意力编码网络，避免了RNN系模型的缺点如难以并行化，需要大量数据等

- Interactive Attention Networks for Aspect-Level Sentiment Classification（IJCAI 2017）<a href="https://arxiv.org/pdf/1709.00893.pdf"> paper</a> 
本文作者认为Aspect-level的情感分类任务中，target与context应该具有交互性，即context应该是target-specific的，target也应该是context-specific的，传统模型中将二者分开建模或只针对其一，本文利用attention实现二者交互

- Learning to Attend via Word-Aspect Associative Fusion for Aspect-based Sentiment Analysis（AAAI 2018）<a https://arxiv.org/abs/1712.05403"> paper</a>
本文作者是针对emnlp2016一篇文章中所提出的AETE-LSTM模型所存在的三个缺点提出的改进模型，提出aspect与word融合层来巧妙地分离各层的职责，使模型首先对aspect和words之间的关系进行建模，然后使注意力层专注于学习已经经过融合的上下文words的相对重要性
