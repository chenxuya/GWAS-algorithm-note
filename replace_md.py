import re

def replace_symbols_in_md(filename):
    # 读取原始文件内容
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # 替换 '\[' 和 '\]' 为 '$$'
    content = re.sub(r'$$', r'\\\[', content)
    content = re.sub(r'$$', r'\\\]', content)
    # 替换 '\(' 和 '\)' 为 '$'
    content = re.sub(r'$', r'\\\(',content)  # 替换 '\(' 为 '$'
    content = re.sub(r'$', r'\\\)',content)  # 替换 '\)' 为 '$'

    # 写回文件，覆盖原文件内容
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


import os

# 获取当前文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取当前文件夹下的所有文件名
file_names = os.listdir(current_dir)

# 遍历文件名列表
for file_name in file_names:
    # 检查文件是否为 Markdown 文件
    if file_name.endswith('.md'):
        # 拼接文件路径
        file_path = os.path.join(current_dir, file_name)

        # 调用 replace_symbols_in_md 函数
        replace_symbols_in_md(file_path)

# filename = r"F:\single cell\GWAS\learning\GWAS_algorithm\papersnote\skip-gram\note.md"
# replace_symbols_in_md(filename)

        print(f"文件 {file_name} 处理完成。")
