"""
this script can identify xml tags competency within a column of selected Dataframe.
It will collect the complete and incomplete tags into 2 separate columns.
"""

import pandas as pd
from lxml import etree
import re

data = {
    'Response': [
        '<tag1>content1</tag1> some text here',
        '<tag2>content2<tag3>content3</tag3></tag2> more text',
        "<tag4 content=\"attribute\">content4</tag4> additional text",
        '<tag5>incomplete</tag5> extra text',
        '</tag6> miscellaneous text',
        # ... 其他数据
    ]
}
# df = pd.DataFrame(data)

# 正则表达式匹配完整的XML标签
complete_tag_pattern = re.compile(r'<(\w+)[^>]*>.*?</\1>')

# 正则表达式匹配不完整的XML标签
incomplete_tag_pattern = re.compile(r'<(\w+)[^>]*>.*?$')

# 正则表达式匹配所有开放标签和闭合标签
open_tag_pattern = re.compile(r'<(\w+)[^>]*>')
close_tag_pattern = re.compile(r'</(\w+)>')


# 函数：使用正则表达式提取标签，并分类完整和不完整的标签
def classify_tags(xml_string):
    # 找到所有开放标签和闭合标签
    open_tags = set(open_tag_pattern.findall(xml_string))
    close_tags = set(close_tag_pattern.findall(xml_string))

    # 确定不完整的标签：只有开放标签或只有闭合标签的
    incomplete_tags = list(open_tags - close_tags) + list(close_tags - open_tags)

    # 确定完整的标签：既有开放标签又有闭合标签的
    complete_tags = [f"<{tag}></{tag}>" for tag in open_tags.intersection(close_tags)]

    # 将标签列表转换为字符串
    complete_tags_str = ', '.join(complete_tags) if complete_tags else ''
    incomplete_tags_str = ', '.join(
        [f"<{tag}>" if tag in open_tags else f"</{tag}>" for tag in incomplete_tags]) if incomplete_tags else ''

    return complete_tags_str, incomplete_tags_str


def check_df_tags(defaultDf=pd.DataFrame(data)):

    # 应用函数到DataFrame
    df['CompleteTags'], df['IncompleteTags'] = zip(*df['Response'].apply(classify_tags))

    return df


# 正则表达式匹配非XML标签的英文字符
non_tag_english_chars_pattern = re.compile(r'>[^<>]*<|<[^<>]*>')


# 函数：提取非XML标签的英文字符
def extract_english_chars(response):
    # 移除XML标签
    non_tags_content = re.sub(non_tag_english_chars_pattern, '', response)

    # 提取英文字符
    english_chars = re.findall(r'[A-Za-z]+', non_tags_content)

    # 将提取的英文字符转换为字符串
    english_chars_str = ' '.join(english_chars) if english_chars else ''

    return english_chars_str


def check_df_english(defaultDf=pd.DataFrame(data)):
    # 应用函数到DataFrame
    df['EnglishChars'] = df['Response'].apply(extract_english_chars)

    return df

if __name__ == '__main__':
    df = pd.read_excel('test_response0529_160times.xlsx')
    checked_df = check_df_tags(df)
    checked_df = check_df_english(checked_df)
    # 保存结果到Excel文件
    checked_df.to_excel('output.xlsx', index=False)

