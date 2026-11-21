from torch import nn

from dm02_encoder_element import (
    MultiHeadAttention,  # 多头注意力层
    FeedForward,  # 前馈全连接层
)
from dm03_encoder_sublayer import SublayerConnection  # 子层连接结构(残差归一化层)


# 解码器子层
class DecoderLayer(nn.Module):
    """
    解码器子层：x -> 掩码多头自注意力层 -> 残差归一化 -> 多头自注意力层 -> 残差归一化 -> 前馈全连接层 -> 残差归一化
    """

    def __init__(self, d_model, n_head=8, d_ff=2048, dropout_p=0.1):
        """
        初始化参数
        :param d_model: 隐藏层维度
        :param n_head: 多头自注意力层头数
        :param d_ff: 前馈全连接层维度数
        :param dropout_p: 随机失活概率
        """
        super().__init__()
        self.d_model = d_model
        self.masked_self_attn = MultiHeadAttention(d_model, n_head=n_head, dropout_p=dropout_p)
        self.src_attn = MultiHeadAttention(d_model, n_head=n_head, dropout_p=dropout_p)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p=dropout_p)
        # 子层连接结构(残差归一化层)
        self.sublayer_connect_0 = SublayerConnection(d_model, dropout_p)
        self.sublayer_connect_1 = SublayerConnection(d_model, dropout_p)
        self.sublayer_connect_2 = SublayerConnection(d_model, dropout_p)

    def forward(self, x,encoder_output, source_mask, target_mask):
        """
        解码器层前向传播
        :param x: 解码器输入序列
        :param encoder_output: 编码器输出
        :param source_mask: 编码器掩码，用于 编码器-解码器 注意力，屏蔽掉 Encoder 输入序列中的 Padding 部分
        :param target_mask: 目标掩码，用于 自注意力
        :return:
        """
        # 掩码多头自注意力层 + 子层连接结构 0
        x = self.sublayer_connect_0(
            x, self.masked_self_attn(query=x, key=x, value=x, mask=target_mask)
        )
        # 多头自注意力层 + 子层连接结构 1
        x = self.sublayer_connect_1(
            x, self.src_attn(query=x, key=encoder_output, value=encoder_output, mask=source_mask)
        )
        # 前馈全连接层 + 子层连接结构 2
        x = self.sublayer_connect_2(x, self.feed_forward(x))

        return x
