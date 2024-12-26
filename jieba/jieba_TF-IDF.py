from jieba import analyse
# 引入TF-IDF关键词抽取接口
tfidf = analyse.extract_tags

# 原始文本
text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
        是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
        线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
        线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
        同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"

# 基于TF-IDF算法进行关键词抽取
# 还是利用cut中的__cut_DAG（精确模式）先对文本进行分词，他这里的IF-IDF值是通过读取idf.txt获取的。然后计算每个词的值(IF-IDF值/total)，最后对值进行排序。
keywords = tfidf(text,withWeight=True)
print("keywords by tfidf:")

# 输出抽取出的关键词
for keyword in keywords:
    print(keyword)