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

- Learning to Attend via Word-Aspect Associative Fusion for Aspect-based Sentiment Analysis（AAAI 2018）<a href = "https://arxiv.org/abs/1712.05403"> paper</a>
本文作者是针对emnlp2016一篇文章中所提出的AETE-LSTM模型所存在的三个缺点提出的改进模型，提出aspect与word融合层来巧妙地分离各层的职责，使模型首先对aspect和words之间的关系进行建模，然后使注意力层专注于学习已经经过融合的上下文words的相对重要性

- SentiLARE:Sentiment-Aware Language Representation Learning with Linguistic Knowledge(2020 emnlp)<br/>
将词的情感极性信息(从外部情感词典中获得)和词的词性信息(序列标注获得)融入到预训练语言模型(Bert)中。具体来说，第一，为了获得句子每个词的情感极性，将句子与情感词典中词的描述做相似度计算，得到词在不同使用情况下的权重，将不同情况下词按照加权求和方式得到最终词的情感极性。第二，使用词向标注工具对词进行标注。第三，将词的情感极性和词性信息作为embedding作为Bert的额外输入。第四，使用词的情感预测，词性预测，句子预测作为bert的预测任务。

- A structure-enhanced graph convolutional network for sentiment analysis(2020 emnlp)<br/>
词与词之间有不同依存关系应该区别对待，GCN应该要能区别出这些信息。论文将词性标注的特征和词与词之间的依存关系特征结合经过变换融入到GCN的邻接矩阵中，这样GCN进行节点消息传递时会考虑到上述两种特征。

- Using the Past Knowledge to Improve Sentiment Classification(2020 emnlp)<br/>
提出一种终生学习(lifelong learning)模型，模型可以保留和选择过去的知识以提升新任务的性能。具体来说，在学习新任务时，将门控机制控制前一个任务参数的流入，使用知识蒸馏的方式指导新任务的学习，直到所有任务学习完毕。

- Label-Consistency based Graph Neural Networks for Semi-supervised Node Classification(acm)<br/>
在半监督学习，GNNs中相邻的节点倾向于有相同的标签。论文提出了label-consistency GNN(LC-GNN),LC-GNN可以聚合那些具有相同标签但在图中不相连的节点之间的信息，扩大了节点的感受野。具体来说，计算节点间标签的相似度并写入邻接矩阵中，这样每个节点将会考虑到与其标签相一致的节点信息。

- Relation-Aware Collaborative Learning for Unified Aspect-Based Sentiment Analysis(2020 acl)<br/>
论文提出建模四种关系：AE任务和OE任务关系R1；R1和SC任务的关系R2；OE任务和SC任务的关系R3；AE任务和SC任务R4。具体来说，利用attention机制建模R1，将R1的attention分数矩阵输入到SC任务中以建模R2，将OE任务的输出输入到SC任务中以建模R3，通过不预测非aspect词的情感极性以建模R4。

- Modelling Context and Syntactical Features for Aspect-based Sentiment Analysis(2020 acl)<br/>
论文将词的dependency embedding,contextualized embedding(bert的输出),pos embedding输入到编码层，利用self-attention进行交互，最后将三种表示拼接作为最终句子表示用于AE任务；在SC任务中，将local context,global context分别建模，使用依存树中的语义距离作为词与词的距离衡量标准对距离过远的词进行mask或者权重降低，让模型更加关注局部的词，最后将局部的表示和全局的表示用self-attention交互送入分类层。

- SentiBERT: A Transferable Transformer-Based Architecture for Compositional Sentiment Semantics<br/>
论文针对复合的情感短语难以判断情感极性的问题进行了研究，因为复合情感短语经常包含情感的反转。论文提出sentibert可以有效捕捉复合的情感短语语义。具体来说，利用短语结构树提取复合情感短语的语义信息，在短语结构树上利用attention建模节点与节点间的关系，通过预测短语节点的情感极性训练bert得到与复合情感短语相关的sentibert。

- Cross-Lingual Unsupervised Sentiment Classification with Multi-View Transfer Learning<br/>
针对情感分析中标注数据有限且不易获得的问题，论文利用多视角的迁移学习做无监督的跨语言情感分析。具体来说，1.利用无监督的机器翻译模型（Encoder Decoder架构）将源语言编码，解码回源语言。2.利用无监督的机器翻译模型（Encoder Decoder架构）翻译成其他语言再翻译回源语言. 3.利用一个语言类别判别器(Lauguage discriminator)对抗训练。1可以提取领域内的信息，2可以提取跨领域信息，3可以拉大两个encoder的距离使得包含的信息是独特的

- Dependency Graph Enhanced Dual-transformer Structure for Aspect-based Sentiment Classification<br/>
对于依存树解析存在的错误和不稳定的问题，情感分析模型在利用依存树时可能会因为依存树解析错误而引起模型提取错误信息。论文通过分开解码两部分信息，一部分是经过transformer的信息，另一部分是加入了依存树的GCN模块，这两部分通过互相加入信息，弥补了某一部分信息的不足，从而即利用了依存树信息，又缓解了依存树解析错误带来的影响。

- SimCSE: Simple Contrastive Learning of Sentence Embeddings<br/>
论文利用对比学习框架用于提升句子句子表示，框架分为无监督方式和有监督方式两种。具体来说，论文发现一个句子经过两次bert可以作为对比学习的正例，一个batch中其他句子为反例，进行对比学习的训练，这是无监督方式。对于有监督方式，论文利用自然语言推理数据集中标注语料中蕴含关系作为正例进行对比学习训练。论文对比其他方法效果有很大提升。

- Jointly Learning Aspect-Focused and Inter-Aspect Relations with Graph Convolutional Networks for Aspect Sentiment Analysis(2020 coling)<br/>
论文利用图卷积神经网络提取依存树中长距离词之间的关系，并且建模了aspect关键词信息，多个aspect间的相互关系。具体来说，通过改变邻接矩阵中aspect关键词的权值达到更加关注aspect关键词的效果；通过在邻接矩阵中加入aspect词与aspect词的边连接达到建模多个aspect间的效果。

- Unsupervised Natural Language Inference via Decoupled Multimodal Contrastive Learning(2020 emnlp)<br/>
论文利用多模态对比学习用于无监督的自然语言推理任务。具体来说，通过一张图片和对应的句子描述的两种表示作为正例进行对比学习，如此句子表示中可以包含丰富的图像信息。进一步，通过图片中的像素块和对应的词两种表示（利用attention机制动态捕捉）作为正例进行对比学习，更加丰富了词的含义。预训练完后，将包含图像信息的文字表示用于自然语言推理任务，取得了很好的效果。

- Graph Ensemble Learning over Multiple Dependency Trees for Aspect-level Sentiment Classification(2021 naacl)<br/>
对于ABSA中用GNN集成依存树存在依存树解析错误的普遍问题，论文将多个解析器的解析结果进行集成，集成方式有两种，一种是所有解析结果的并集，另一种是交集。这使得模型更加鲁棒性。

- Aspect-based Sentiment Analysis with Type-aware Graph Convolutional Networks and Layer Ensemble(2021 naacl)<br/>
论文在集成依存树时，不仅考虑了词与词之间是否依存连接关系，还考虑了词之间的依存关系的类别。论文还提出在GCN层之间做attention以利用不同GCN层之间的不同语义关系。具体来说，使用adjcent matrix(邻接矩阵)和 relaition matrix（关系矩阵）代表句子中词之间的邻接关系和关系的类别，在GCN节点进行消息传递时集成关系矩阵生成词与词之间的attention分数，这样每个词在集成其他词信息时都考虑到关系类别信息，最后在GCN多层之间做attention得到最后的表示用于分类。

- Selective Attention Based Graph Convolutional Networks for Aspect-Level Sentiment Classification(2021 naacl)<br/>
有些句子aspect与情感词opinion在依存树中会超过两次才能到达，意味着需要超过两层的GCN才能建模这种远距离关系，然鹅现有的GCN层数一般两层就会达到性能饱和，论文针对这个问题，将图改为完全图利用多头attenion建模任意词与词之间的关系，通过top-k方式选择k个最重要的上下文，忽略其他词。

- Understanding Pre-trained BERT for Aspect-based Sentiment Analysis（2020 coling）<br/>
论文探究预训练bert的MLM任务是否能够含有ABSA任务所需要的一些特征。如aspect词能够包含opinion的信息，aspect能包含情感倾向信息，opinion词可以反映aspect词的情感极性。通过多个实验发现，预训练bert包含丰富的领域信息和本身aspect的语义信息，所以预训练bert并不能很好建模aspect和opion词的关系，但是对于aspect抽取任务是有效的，因为aspect抽取任务特别需要领域特征和需要知道词是不是aspect的特征，这些MLM能够满足。论文提出：需要一个更加适合的预训练任务能够解耦aspect和opinion的信息，以及使得aspect具有情感信息的任务(这或许是个新的可开发的领域，可以试试)。
