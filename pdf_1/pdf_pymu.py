import pymupdf # imports the pymupdf library
import re
import opencc

def remove_single_char_lines(text):
    # 使用正则表达式匹配形如 \nX\n 的模式，其中 X 是单个字符
    return re.sub(r'\n(.)\n', lambda m: '\n' if len(m.group(1)) == 1 else m.group(0), text)

def remove_extra_spaces(text):
    # 使用正则表达式替换多余的空格为单个空格
    return re.sub(r'\s+', ' ', text).strip()

def format_text(text):
    """
    This function removes newline characters from the text and segments it based on periods.
    It then adds a newline character after each segment.
    """
    # Remove newline characters
    text_without_newlines = text.replace('\n', '')
    
    # Segment the text based on periods
    segments = text_without_newlines.split('。')
    
    # Add a newline character after each segment
    formatted_text = '。\n'.join(segments)
    
    return formatted_text


def remove_lines_without_any_punctuation(text):
    # 标点符号的正则表达式，包括中文和英文标点
    punctuation = r'[\u3000-\u303F.,;!?，。；！？]'
    
    # 按行分割文本
    lines = text.split('\n')
    
    lines = [remove_extra_spaces(line) for line in lines]
    # 过滤掉不包含任何标点符号的行
    filtered_lines = [line for line in lines if re.search(punctuation, line)]
    
    # 将过滤后的行重新组合成字符串
    return '\n'.join(filtered_lines)

# 判断文本是否是繁体
def is_traditional(text):
    # 这里可以根据实际需求定制检测逻辑，简单的方法是检查字符集
    # 如果存在大量繁体字符，则判定为繁体
    # 你可以根据实际需求调整规则，例如通过正则表达式来检查
    traditional_range = r'[\u4e00-\u9fff\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F]'
    return bool(re.search(traditional_range, text))

# 创建繁体转简体的转换器
cc = opencc.OpenCC('t2s')  # 繁体转简体

def extract_text_with_formatting(doc):
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_instances = page.get_text("dict")
        
        for block in text_instances['blocks']:
            if block['type'] == 0:  # 0 表示文本块
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text']
                        if is_traditional(text):
                            text = cc.convert(text)
                        
                        text = remove_single_char_lines(text)
                        text = remove_lines_without_any_punctuation(text)
                        text = format_text(text)
                        
                        full_text += text + "\n"
    
    return full_text.strip()

doc = pymupdf.open("E:\\vscode_test\\data\\山西汾酒_2012年度社会责任报告.pdf") # open a document
formatted_text = extract_text_with_formatting(doc)
print(formatted_text)
