import math

import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """

        :param vocab_size: 词典大小
        :param embedding_dim: 词向量维度
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # 定义词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        # 将 词向量的数值 乘上 词向量维度大小的开平方
        return self.embedding(x) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p=0.1, max_len=5000):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_p)
        self.max_len = max_len

        # 建一个全0的矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 位置序号 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 核心计算：1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数位置用 sin，奇数位置用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个 batch 维度：[1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为 buffer，这样它不会作为参数被优化器更新，但会跟着模型保存和加载，并能自动转移到 GPU
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # 截取对应的长度加到输入上
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

# 整合输入部分
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout_p=0.1,max_len=5000):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout_p=dropout_p, max_len=max_len)

    def forward(self, x):
        x = self.embedding(x)   # 词嵌入
        x = self.positional_encoding(x) # 位置编码
        # 返回数据
        return x




def dm_test_embedding():
    vocab_size = 128
    embedding_dim = 64
    embd = Embedding(vocab_size, embedding_dim)
    x = torch.randint(0, vocab_size, size=(5, 10))
    res = embd(x)
    print(res.shape)
    print(res)


def use_positional_encoding():
    vocab_size = 1000
    d_model = 512
    # 词嵌入层
    embd = Embedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, size=(3, 5))
    embd_x = embd(x)
    # 位置编码层
    pe = PositionalEncoding(d_model)
    pe_x = pe(embd_x)

    return pe_x


if __name__ == "__main__":
    # dm_test_embedding()
    use_positional_encoding()
