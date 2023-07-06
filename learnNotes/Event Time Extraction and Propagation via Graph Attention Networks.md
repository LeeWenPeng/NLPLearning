# Event Time Extraction and Propagation via Graph Attention Networks

## 1 论文简介

1.   cite：Haoyang Wen, Yanru Qu, Heng Ji, Qiang Ning, Jiawei Han, Avi Sil, Hanghang Tong, and Dan Roth. 2021. [Event Time Extraction and Propagation via Graph Attention Networks](https://aclanthology.org/2021.naacl-main.6). In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 62–73, Online. Association for Computational Linguistics.
2.   paper：<https://aclanthology.org/2021.naacl-main.6/>
3.   code：<https://github.com/wenhycs/naacl2021-event-time-extraction-and-propagation-via-graph-attention-networks>

## 2 解决了什么问题？

为事件确认精确的时间轴

## 3 使用了什么方法

### 3.1 基于实体槽位填充的四元组时序表示方法

采用基于实体槽位填充的四元组时序表示方法，在给定整个文档的情况下，预测最早\最晚可能的开始\结束的时间

事件不确定的时间边界通常可以通过文档的全局上下文的相关事件推断出

### 基于图注意力机制网络，在有共享论元和时间关系构造的文档级事件图上传播时间信息

## 4 取得了什么成果

1.   与上下文嵌入法相比，匹配率上获得7.0%绝对增益
2.   与句子级手工事件关系论元注释相比，匹配率提高了16.3%

1.   使用了什么数据集

     ACE2005

