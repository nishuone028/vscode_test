from collections import defaultdict, Counter
import re
import random

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(Counter)

    def train(self, sentences):
        for sentence in sentences:
            words = self.tokenize(sentence)
            for i in range(len(words) - self.n + 1):
                ngram = tuple(words[i:i+self.n])
                prefix = ngram[:-1]
                suffix = ngram[-1]
                self.model[prefix][suffix] += 1

    def tokenize(self, text):
        # 使用正则表达式进行简单的分词，这里假设空格分隔单词
        return re.findall(r'\b\w+\b', text.lower())

    def get_probability(self, context, word):
        if not context:
            return self.model[()][word] / sum(self.model[()].values())
        context_counter = self.model[tuple(context[-(self.n-1):])]
        return context_counter[word] / sum(context_counter.values())

    def generate(self, context, num_words):
        output = list(context)
        for _ in range(num_words):
            if not context:
                context = ()
            else:
                context = tuple(context)  # 将 context 转换为元组
            probabilities = self.model[context]
            if not probabilities:
                break
            word = self.sample_from(probabilities)
            output.append(word)
            context = tuple(output[-(self.n-1):])
        return ' '.join(output)

    def sample_from(self, distribution):
        words, weights = zip(*distribution.items())
        return self.weighted_choice(words, weights)

    @staticmethod
    def weighted_choice(words, weights):
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for c, w in zip(words, weights):
            if upto + w >= r:
                return c
            upto += w

# 示例使用
sentences = [
    "I like to eat apples",
    "I like to eat bananas",
    "I like to eat apples and bananas"
]

ngram_model = NGramModel(n=2)
ngram_model.train(sentences)

# 获取给定上下文后一个词的概率
context = ["to", "eat"]
word = "apples"
print(f"Probability of '{word}' after '{' '.join(context)}': {ngram_model.get_probability(context, word)}")

# 使用模型生成文本
print("Generated text:", ngram_model.generate(context, 4))
