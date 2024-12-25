# from pdfminer.high_level import extract_text



# text = extract_text("E:\\vscode_test\\data\\晨鸣纸业_2009年度社会责任报告.pdf")
# print(text)

from io import StringIO
from pdfminer.high_level import extract_text_to_fp,extract_pages
from pdfminer.layout import LAParams,LTTextContainer

output_string = StringIO()
with open('E:\\vscode_test\\data\\中国中冶_2014年度社会责任报告.pdf', 'rb') as fin:
    extract_text_to_fp(fin, output_string, laparams=LAParams(),
                       output_type='html', codec=None)
print(output_string.getvalue().strip())


for page_layout in extract_pages("E:\\vscode_test\\data\\中国中冶_2014年度社会责任报告.pdf"): 
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            print(element.get_text())