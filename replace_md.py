import re

def replace_symbols_in_md(filename):
    # 读取原始文件内容
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # 替换 '\[' 和 '\]' 为 '$$'
    content = re.sub(r'\\\[', r'$$', content)
    content = re.sub(r'\\\]', r'$$', content)
    # 替换 '\(' 和 '\)' 为 '$'
    content = re.sub(r'\\\(', r'$', content)  # 替换 '\(' 为 '$'
    content = re.sub(r'\\\)', r'$', content)  # 替换 '\)' 为 '$'

    # 写回文件，覆盖原文件内容
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

# 提示用户输入文件名
filename = r"F:\single cell\GWAS\learning\GWAS_algorithm\papersnote\brnn.md"
replace_symbols_in_md(filename)

print(f"文件 {filename} 处理完成。")
