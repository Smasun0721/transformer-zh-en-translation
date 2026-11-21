import torch
from torch import nn

from dm02_encoder_element import LayerNorm
from dm06_decoder_layer import DecoderLayer


class Decoder(nn.Module):
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
            [DecoderLayer(d_model, n_head, d_ff, dropout_p) for _ in range(n_layer)]
        )
        # 最终的规范化层 (标准 Post-LN 结构通常包含这一步)
        self.norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)

        return self.norm(x)


if __name__ == "__main__":
    # 1. 模拟输入
    d_model = 512
    batch_size = 3
    tgt_seq_len = 4

    # 假设 Encoder 的输出 (来自你之前的 Encoder 模块)
    encoder_output = torch.randn(batch_size, 5, d_model)
    # 假设 Decoder 的输入 (通常是目标序列的词嵌入+位置编码)
    tgt_x = torch.randn(batch_size, tgt_seq_len, d_model)

    # 2. 实例化 Decoder
    decoder = Decoder(d_model=d_model, n_head=8, d_ff=2048, n_layer=6)

    # 3. 前向传播 (暂不传 mask)
    output = decoder(tgt_x, encoder_output)

    print("Decoder 输出形状:", output.shape)
    # 预期输出: torch.Size([3, 4, 512]) -> (batch_size, tgt_seq_len, d_model)