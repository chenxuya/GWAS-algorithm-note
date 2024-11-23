import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re

# 假设文档路径为 'corpus.txt'
file_path = './Prince2.txt'

# Step 1: 读取文档
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Step 2: 文本清洗
# 移除标点符号并将文本转为小写
text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
text = text.lower()  # 转为小写

# Step 3: 分词
words = text.split()
print(f"总词数：{len(words)}")

# Step 4: 构建词汇表
vocab = list(set(words))  # 获取独特的词汇
word2idx = {word: idx for idx, word in enumerate(vocab)}  # 词到索引映射
idx2word = {idx: word for word, idx in word2idx.items()}  # 索引到词的映射

print(f"词汇表大小：{len(vocab)}")

# Step 5: 生成 (中心词, 上下文词) 对
context_window = 2  # 上下文窗口大小

# 生成所有 (中心词, 上下文词) 对
data = []
for i, word in enumerate(words):
    center_word_idx = word2idx[word]
    # 在窗口范围内选择上下文词
    for j in range(max(0, i - context_window), min(len(words), i + context_window + 1)):
        if i != j:
            context_word_idx = word2idx[words[j]]
            data.append((center_word_idx, context_word_idx))

# 打印部分 (中心词, 上下文词) 对
print("样本数据对 (中心词, 上下文词):", data[:5])
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        # 输入向量矩阵 V
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 输出向量矩阵 U
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, center_words, context_words):
        # 中心词和上下文词的向量表示
        center_embeds = self.input_embeddings(center_words)  # (batch_size, embedding_dim)
        context_embeds = self.output_embeddings(context_words)  # (batch_size, embedding_dim)
        
        # 计算向量内积
        scores = torch.sum(center_embeds * context_embeds, dim=1)
        return scores

# 参数设置
embedding_dim = 10  # 词向量维度
learning_rate = 0.01
epochs = 100
negative_samples = 3  # 每个正样本生成的负样本数

# 初始化模型和优化器
model = SkipGram(vocab_size=len(vocab), embedding_dim=embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_function = nn.BCEWithLogitsLoss()  # 使用负采样损失

# 训练过程
for epoch in range(epochs):
    total_loss = 0
    for center_word, context_word in data:
        # 生成负样本
        negative_contexts = np.random.choice(len(vocab), negative_samples, replace=False)
        
        # 正样本标签为 1，负样本标签为 0
        positive_target = torch.tensor([context_word], dtype=torch.long)
        negative_targets = torch.tensor(negative_contexts, dtype=torch.long)
        
        # 中心词
        center_input = torch.tensor([center_word], dtype=torch.long)
        
        # 正样本损失
        optimizer.zero_grad()
        positive_scores = model(center_input, positive_target)
        positive_loss = loss_function(positive_scores, torch.ones_like(positive_scores))
        
        # 负样本损失
        negative_scores = model(center_input, negative_targets)
        negative_loss = loss_function(negative_scores, torch.zeros_like(negative_scores))
        
        # 总损失
        loss = positive_loss + negative_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data)}")

# 提取词向量
input_embeddings = model.input_embeddings.weight.data.numpy()
output_embeddings = model.output_embeddings.weight.data.numpy()

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2)
reduced_embeddings = tsne.fit_transform(input_embeddings)

# 可视化
plt.figure(figsize=(8, 6))
for i, label in enumerate(vocab):
    x, y = reduced_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,5), ha='center')
plt.show()
