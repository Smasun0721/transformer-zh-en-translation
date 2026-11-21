from torch import nn
import dm01_input
from dm02_encoder_element import (
    MultiHeadAttention,  # 多头注意力层
    FeedForward,  # 前馈全连接层
    LayerNorm,  # 规范化层
)
from dm03_encoder_sublayer import SublayerConnection  # 子层连接结构(残差归一化层)


# 编码器子层
class EncoderLayer(nn.Module):
    """
    编码器子层：x -> 多头自注意力层 -> 残差归一化 -> 前馈全连接层 -> 残差归一化
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
        self.self_attn = MultiHeadAttention(d_model, n_head=n_head, dropout_p=dropout_p)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p=dropout_p)
        # 子层连接结构(残差归一化层)
        self.sublayer_connect_1 = SublayerConnection(d_model, dropout_p)
        self.sublayer_connect_2 = SublayerConnection(d_model, dropout_p)

    def forward(self, x, mask=None):
        # 多头自注意力层 + 子层连接结构 1
        x = self.sublayer_connect_1(
            x, self.self_attn(query=x, key=x, value=x, mask=mask)
        )
        # 前馈全连接层 + 子层连接结构 2
        x = self.sublayer_connect_2(x, self.feed_forward(x))

        return x


if __name__ == "__main__":
    encoder_layer = EncoderLayer(d_model=512, n_head=8, d_ff=2048, dropout_p=0.1)
    pe_x = dm01_input.use_positional_encoding()
    result = encoder_layer(pe_x)
    print(result)
    print(result.shape)
    print(type(result))
    print()
    print(encoder_layer)
