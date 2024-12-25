class MMSEG:
    def __init__(self, dictionary):
        self.dictionary = set(dictionary)

    def max_match(self, text):
        # 从self.dictionary 中找到最长的词条，并将这个最长词条的长度赋值给 max_length。
        max_length = len(max(self.dictionary, key=len))
        result = []
        index = 0

        while index < len(text):
            for end in range(min(index + max_length, len(text)), index, -1):
                word = text[index:end]
                if word in self.dictionary:
                    result.append(word)
                    index = end
                    break
            else:
                # 如果没有匹配到词语，则按单字切分
                result.append(text[index])
                index += 1
        return result

# 示例使用
dictionary = ['研究', '生命', '科学', '研究生', '命', '的', '起源']
mmseg = MMSEG(dictionary)
text = '研究生命的起源'
print(mmseg.max_match(text))
