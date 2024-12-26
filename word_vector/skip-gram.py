# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# 1 下载语料文件，并校验文件字节数是否正确
url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if (statinfo.st_size == expected_bytes):
        print("get text and verified")
    else:
        raise Exception("text size is not correct")

    return filename

filename = maybe_download("text8.zip", 31344016)


# 2 语料处理，弄成一个个word组成的list, 以空格作为分隔符。
# 如果是中文语料，这一步还需要进行分词
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulay = read_data(filename)
print("total word size %d" % len(vocabulay))
print("100 words at first: ", vocabulay[0:100])

# 3 词表制作，根据出现频率排序，序号代表这个单词。词语编码的一种常用方式
def build_dataset(words, n_words):
    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionay = dict()
    for word, _ in count:
        # 利用按照出现频率排序好的词语的位置，来代表这个词语
        dictionay[word] = len(dictionay)

    # data包含语料库中的所有词语，低频的词语标注为UNK。这些词语都是各不相同的
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionay:
            index = dictionay[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count   # unk的个数

    # 将key value reverse一下，使用数字来代表这个词语
    reversed_dictionary = dict(zip(dictionay.values(), dictionay.keys()))
    return data, count, dictionay, reversed_dictionary

VOC_SIZE = 50000
data, count, dictionary, reversed_dictionary = build_dataset(vocabulay, VOC_SIZE)
del vocabulay
print("most common words", count[0:5])
# 打印前10个单词的数字序号
print("sample data", data[:10], [reversed_dictionary[i] for i in data[:10]])

# 4 生成训练的batch label对
data_index = 0
# skip_window表示与target中心词相关联的上下文的长度。整个Buffer为 (2 * skip_window + 1)，从skip_window中随机选取num_skips个单词作为label
# 最后形成 target-&gt;label1 target-&gt;label2的batch label对组合
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # 将skip_window的数据组合放入Buffer中
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)   # 防止超出data数组范围，因为batch可以取很多次迭代。所以可以循环重复

    # num_skips表示一个Buffer中选取几个batch-&gt;label对，每一对为一个batch，故需要batch_size // num_skips个Buffer
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]

        # 一个Buffer内部寻找num_skips个label
        for j in range(num_skips):
            # 寻找label的位置，总共会有num_skips个label
            while target in targets_to_avoid:   # 中间那个为batch，不能选为target.也不能重复选target
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)

            # 中心位置为batch，随机选取的num_skips个其他位置的为label
            batch[i * num_skips + j] = buffer[skip_window]  #
            labels[i * num_skips + j, 0] = buffer[target]   # 遍历选取的label

        # 一个Buffer内的num_skips找完之后，向后移动一位，将单词加入Buffer内，并将Buffer内第一个单词移除，从而形成新的Buffer
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # 所有batch都遍历完之后，重新调整data_index指针位置
    data_index = (data_index + len(data) - span) % len(data)

    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[1], reversed_dictionary[batch[i]], "-&gt;", labels[i, 0], reversed_dictionary[labels[i, 9]])

# 5 构造训练模型
batch_size = 128
embedding_size = 128  # 词向量为128维，也就是每一个word转化为的vec是128维的
skip_window = 1   # 滑窗大小为1， 也就是每次取中心词前后各一个词
num_skips = 2     # 每次取上下文的两个词

# 模型验证集, 对前100个词进行验证，每次验证16个词
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# 噪声词数量
num_sampled = 64

graph= tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)     # 验证集

    with tf.device("/cpu:0"):
        # 构造embeddings, 50000个词语，每个词语为长度128的向量
        embeddings = tf.Variable(tf.random_uniform([VOC_SIZE, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(tf.truncated_normal([VOC_SIZE, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([VOC_SIZE]))

    # 利用nce loss将多分类问题转化为二分类问题，从而使得词向量训练成为可能，不然分类会是上万的量级
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,       # inputs为经过embeddings词向量之后的train_inputs
            num_sampled=num_sampled,    # 噪声词
            num_classes=VOC_SIZE,
        )
    )
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 归一化embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()


# 6 训练
num_steps = 100000
with tf.Session(graph=graph) as session:
    init.run()

    average_loss = 0
    for step in xrange(num_steps):
        # 构建batch，并进行feed
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

       # run optimizer和loss，跑模型
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0 and step &gt; 0:
            average_loss /= 2000
            print("average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # 1万步，验证一次
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1: top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()

