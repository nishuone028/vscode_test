from jieba import analyse
# 引入TextRank关键词抽取接口
textrank = analyse.textrank


# 先将文本进行分词和词性标注，将特定词性的词（比如名词）作为节点添加到图中。
# 出现在一个窗口中的词语之间形成一条边，窗口大小可设置为2~10之间，它表示一个窗口中有多少个词语。
# 对节点根据入度节点个数以及入度节点权重进行打分，入度节点越多，且入度节点权重大，则打分高。
# 然后根据打分进行降序排列，输出指定个数的关键词。

# 原始文本
text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
        是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
        线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
        线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
        同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"

print("\nkeywords by textrank:")
# 基于TextRank算法进行关键词抽取
keywords = textrank(text,withWeight=True)
# 输出抽取出的关键词
for keyword in keywords:
    print(keyword)
