# Sentiment Analysis

- Xu 等, 2020 - Aspect Sentiment Classification with Aspect-Specific Opinion Spans(EMNLP2020)<a href="https://www.aclweb.org/anthology/2020.emnlp-main.288/"> paper</a><a href="https://github.com/xuuuluuu/Aspect-Sentiment-Classification"> code</a><br/>
论文基于结构化attention模型，提出了多个CRF用来建模aspect的观点跨度信息。具体来说，加入一个latent label z属于{Yes,No}表示每个词是否在 aspect跨度之内，
再用多个CRF提取aspect的观点跨度信息。

- xie和li等，2018-Aspect Based Sentiment Analysis with Gated Convolutional Networks(acl 2018)<a href="https://www.aclweb.org/anthology/P18-1234/"> paper</a><a href=" https://github.com/wxue004cs/GCAE"> code</a><br/>
论文采用卷积神经网络和门控制机制比之前的lstm+注意力机制模型简洁、运算效率高，门控机制的作用可以在给定的aspect信息中，有选择的提取aspect情感信息
