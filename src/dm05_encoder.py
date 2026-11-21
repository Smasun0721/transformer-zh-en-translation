import torch
from torch import nn

import dm01_input
from dm02_encoder_element import LayerNorm
from dm04_encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
    编码器层 * N
    """

    def __init__(self, d_model, n_head=8, d_ff=2048, dropout_p=0.1, n_layer=6):
        """
        初始化参数
        :param d_model: 隐藏层维度
        :param n_head: 多头自注意力层头数
        :param d_ff: 前馈全连接层维度数
        :param dropout_p: 随机失活概率
        :param n_layer: 编码器层数量
        """
        super().__init__()
        # 编码器层列表
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, d_ff, dropout_p) for _ in range(n_layer)]
        )
        # 最终的规范化层 (标准 Post-LN 结构通常包含这一步)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.norm(x)


# 测试
if __name__ == "__main__":
    encoder = Encoder(d_model=512, n_head=8, d_ff=2048, dropout_p=0.1, n_layer=6)
    pe_x = dm01_input.use_positional_encoding()
    result = encoder(pe_x)
    print(result)
    print(result.shape)
    print(type(result))
    print()
    print(encoder)
