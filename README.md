# sentiment-analysis

Xu 等。 - 2020 - Aspect Sentiment Classification with Aspect-Specific Opinion Spans<a href="https://www.aclweb.org/anthology/2020.emnlp-main.288/">论文地址</a><br/>
论文基于结构化attention模型，提出了多个CRF用来建模aspect的观点跨度信息。具体来说，加入一个latent label z属于{Yes,No}表示每个词是否在 aspect跨度之内，
再用多个CRF提取aspect的观点跨度信息。
