# abstract:

知识图在过去的几年里越来越受欢迎，这要归功于它们在日常搜索引擎中的应用。通常，它们由关于个人和组织的相当静态和百科全书式的事实组成--例如名人的出生日期、职业和家庭成员--从Freebase或维基百科等大型库中获得。本文提出了一种从新闻文章中自动生成知识图的方法和工具。由于新闻文章通过报道事件来描述世界的变化，我们提出了一种使用最先进的自然语言处理和语义网技术来创建事件中心知识图（ECKG）的方法。这样的ECKG捕获关于数十万个实体的长期发展和历史，并且是对传统知识图中的静态百科全书式信息的补充。我们描述了我们的以事件为中心的表示模式、从新闻中提取事件信息的挑战、我们的开源管道以及我们从四个不同的新闻语料库中提取的知识图：综合新闻（Wikinews）、FIFA世界杯、全球汽车工业和空中客车A380飞机。此外，我们还对该流水线提取知识图三元组的准确性进行了评估。此外，通过以事件为中心的浏览器和可视化工具，我们展示了以事件为中心的新闻信息获取方式如何增加用户对领域的理解，促进新闻故事情节的重构，并能够对新闻隐藏事实进行探索性调查。

# 2、background and related work

从新闻文章中自动提取事实和事件的工具

开放信息提取系统(Open Information Extraction System)

+ TextRunner
+ NELL
+ SRL(Semantic Role Labeling)语义角色标记
  + H. Llorens, E. Saquete, B. Navarro, Tipsem (english and spanish): Evaluating
    crfs and semantic roles in tempeval-2, in: Proceedings of the 5th
    International Workshop on Semantic Evaluation, SemEval’10, Association for
    Computational Linguistics, Stroudsburg, PA, USA, 2010, pp. 284–291. URL
    http://dl.acm.org/citation.cfm?id=1859664.1859727.
+ TimeML使用语义角色来提取事件及其关系
+ XLike





