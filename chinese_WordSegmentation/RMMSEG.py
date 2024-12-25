class RMMSEG:
    def __init__(self, dictionary):
        self.dictionary = set(dictionary)
        self.max_length = len(max(self.dictionary, key=len))

    def reverse_max_match(self, text):
        result = []
        index = len(text)

        while index > 0:
            for start in range(max(0, index - self.max_length), index):
                word = text[start:index]
                if word in self.dictionary:
                    result.insert(0, word)  # 插入到结果列表的开头
                    index = start
                    break
            else:
                # 如果没有匹配到词语，则按单字切分
                result.insert(0, text[index - 1])
                index -= 1
        return result

# 示例使用
dictionary = ['研究', '生命', '科学', '研究生', '命', '的', '起源']
rmmseg = RMMSEG(dictionary)
text = '研究生命的起源'
print(rmmseg.reverse_max_match(text))
