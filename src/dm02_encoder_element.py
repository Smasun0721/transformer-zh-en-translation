import math

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import dm01_input


# 获取上/下三角矩阵
def get_triu(row_len, col_size, k=0, reverse=False):
    grid = torch.tensor(np.triu(np.ones((1, row_len, col_size)), k=k), dtype=torch.long)
    return 1 - grid if reverse else grid


# 注意力层
def attention(query, key, value, mask=None, dropout=None):
    """
    注意力计算
    :param query: 查询张量，形状通常为 [batch_size, seq_len, d_model]
    :param key: 键张量，形状通常为 [batch_size, seq_len, d_model]
    :param value: 值张量，形状通常为 [batch_size, seq_len, d_model]
    :param mask: 掩码张量，形状通常与scores一致 [seq_len, seq_len]
    :param dropout: 随机失活，防止过拟合
    :return:
    """

    # 1.定义查询张量Q的特征维度
    d_k = query.size(-1)
    # 2.计算原始注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 3.掩码处理
    # 3.掩码处理
    if mask is not None:
        # 动态获取当前 scores 数据类型的最小值，防止 float16 溢出
        min_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask == 0, min_value)
    # 4.计算注意力权重
    p_attn = F.softmax(scores, dim=-1)
    # 5.随机失活
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 6.计算最终的注意力输出：加权求和
    return torch.matmul(p_attn, value), p_attn


# 模块复制函数
def clones(module, N):
    """
    克隆指定的模块N次
    :param module: 要克隆的模块
    :param N: 克隆的个数
    :return: 模块列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout_p=0.1):
        """
        参数初始化
        :param d_model: 词向量维度
        :param n_head: 多头数量
        :param dropout_p: 随机失活概率
        """
        super().__init__()
        # 确保分头能整除
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.d_k = d_model // n_head  # 每个头的词向量维度
        # 定义4个线性层，前三个用于QKV的线性投影，最后一个用于多头注意力的输出
        self.linears = clones(nn.Linear(d_model, n_head * self.d_k), N=4)
        # 随机失活层
        self.dropout = nn.Dropout(p=dropout_p)
        # 注意力权重
        self.attn = None

    def forward(self, query, key, value, mask=None):
        # 1.判断当前是否有掩码
        if mask is not None:
            mask = mask.unsqueeze(1)  # 增加 batch_size 维

        # 2.获取batch_size大小
        batch_size = query.size(0)

        # 3.线性投影 -> 多头
        query, key, value = [
            linear(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        # 4.多头注意力计算
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 5.将多头注意力结果合并
        attn_x = (
            x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        )

        # 6.多头注意力的输出
        return self.linears[-1](attn_x)


# 前馈全连接层
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        """
        参数初始化
        :param d_model: 隐藏层维度
        :param d_ff: 升维目标维度
        :param dropout_p: 随机失活概率
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear1 = nn.Linear(d_model, d_ff)  # 升维
        self.linear2 = nn.Linear(d_ff, d_model)  # 降维

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


# 规范化层  （标准化）
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """
        参数初始化
        :param d_model: 词向量维度
        :param eps: 小常数，防止分母为0
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # 缩放
        self.bias = nn.Parameter(torch.zeros(d_model))  # 平移

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias  # 标准化


def dm_test_triu():
    grid = get_triu(row_len=5, col_size=5, k=0)
    print(grid)
    print(grid.shape)
    arr = torch.randint(0, 10, (5, 5))
    print(arr)
    masked1 = torch.masked_fill(arr.float(), grid == 0, 0)
    masked2 = torch.masked_fill(arr.float(), grid == 0, -1e9)
    print(masked1)
    print(torch.softmax(masked1, dim=-1))
    print(masked2)
    print(torch.softmax(masked2, dim=-1))


def dm_test_attention():
    pe_x = dm01_input.use_positional_encoding()
    query = key = value = pe_x
    result, p_attn = attention(query, key, value, mask=None, dropout=None)
    print(result)
    print(p_attn)


def use_multihead_attention():
    pe_x = dm01_input.use_positional_encoding()
    query = key = value = pe_x
    multihead_attention = MultiHeadAttention(d_model=512, n_head=8, dropout_p=0.1)
    result = multihead_attention(query, key, value)

    # print(multihead_attention.attn)
    # print(result)
    # print(result.shape)

    return result


def use_ff():
    # 1.获取多头注意力计算结果
    attn_x = use_multihead_attention()
    # 2.获取全连接层对象
    ff = FeedForward(d_model=512, d_ff=2048, dropout_p=0.1)
    # 3.全连接层处理
    ff_x = ff(attn_x)
    # print(ff_x.shape)
    return ff_x


def use_layer_norm():
    # 获取前馈全连接层的结果
    ff_x = use_ff()

    layer_norm = LayerNorm(d_model=512, eps=1e-6)
    ln_x = layer_norm(ff_x)
    # print(ln_x)
    # print(ln_x.shape)
    return ln_x


if __name__ == "__main__":
    use_layer_norm()
