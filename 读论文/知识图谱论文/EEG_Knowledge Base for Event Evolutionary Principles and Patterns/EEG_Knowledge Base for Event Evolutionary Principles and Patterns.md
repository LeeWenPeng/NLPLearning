# 1 Introduction

大量的努力致力于从文本中提取时间和因果关系。TimeBank语料库[15]作为最常用的语料库，已经被大量的时间关系抽取研究所采用。Mani等人[9]应用时间传递性规则极大地扩展了语料库。Chambers等人[4]使用之前学习的事件属性对时间关系进行分类。在因果关系提取方面，Zhao等[19]提取了多种特征来识别同一句子中两个事件之间的因果关系。Radinsky等人[16]通过预先定义的因果模板从大量新闻标题中自动提取因果对，然后利用它们对新闻事件进行预测。然而，这些研究都存在一定的局限性。首先，这一行只能从单个句子中提取关系。其次，这些研究是基于特定上下文的语义来提取关系，而不是从大规模用户生成的文档中发现事件演变的潜在模式。



[4] Chambers, N., Wang, S., Jurafsky, D.: Classifying temporal relations between
events. In: ACL, pp. 173–176. ACL (2007)

19. Zhao, S., Liu, T., Zhao, S., Chen, Y., Nie, J.Y.: Event causality extraction based
    on connectives analysis. Neurocomputing 173, 1943–1950 (2016)

16. Radinsky, K., Davidovich, S., Markovitch, S.: Learning causality for news events
prediction. In: WWW, pp. 909–918. ACM (2012)





EEG 两个关键问题

+ 识别每两个事件之间的关系
+ 区分事件之间各个关系的方向

使用分类框架解决



# 2 Related Work

2.1 Statistical Script