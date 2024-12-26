# gensim是自然语言处理的一个重要Python库，它包括了Word2vec
import gensim
from gensim.models import word2vec

# 语句，由原始语句经过分词后划分为的一个个词语
sentences = [['网商银行', '体验', '好'], ['网商银行','转账','快']]

# 使用word2vec进行训练
# min_count: 词语频度，低于这个阈值的词语不做词向量
# size:每个词对应向量的维度，也就是向量长度
# workers：并行训练任务数
model = word2vec.Word2Vec(sentences, size=256, min_count=1)

# 保存词向量模型，下次只需要load就可以用了
model.save("word2vec_atec")
