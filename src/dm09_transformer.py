import torch
from torch import nn

from dm01_input import TransformerEmbedding  # 输入模块
from dm05_encoder import Encoder  # 编码器
from dm07_decoder import Decoder  # 解码器
from dm08_output import Generator  # 输出模块


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        target_vocab_size,
        d_model=512,
        n_head=8,
        n_layer=6,
        d_ff=2048,
        dropout_p=0.1,
    ):
        """
        参数初始化
        :param src_vocab_size: 源词典大小
        :param target_vocab_size: 目标词典大小
        :param d_model: 隐藏层维度
        :param n_head: 多头注意力头数
        :param n_layer: 编码器、解码器层数
        :param d_ff: 前馈网络（FFN）内部隐藏层维度
        :param dropout_p: 随机失活概率
        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.target_vocab_size = target_vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        # 输入模块(原始输入)
        self.transformer_src_embedding = TransformerEmbedding(
            vocab_size=src_vocab_size,
            d_model=d_model,
            dropout_p=dropout_p,
            max_len=5000,
        )
        # 输入模块(目标输出)
        self.transformer_target_embedding = TransformerEmbedding(
            vocab_size=target_vocab_size,
            d_model=d_model,
            dropout_p=dropout_p,
            max_len=5000,
        )
        # 编码器
        self.encoder = Encoder(
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            d_ff=d_ff,
            dropout_p=dropout_p,
        )
        # 解码器
        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            d_ff=d_ff,
            dropout_p=dropout_p,
        )
        # 输出模块
        self.generator = Generator(
            d_model=d_model,
            vocab_size=target_vocab_size,
        )

    def forward(self, src, target, source_mask=None, target_mask=None):
        # 输入模块(原始输入)
        x = self.transformer_src_embedding(src)
        # 输入模块(目标输出)
        y = self.transformer_target_embedding(target)
        # 编码器
        encoder_output = self.encoder(x, mask=source_mask)
        # 解码器
        decoder_output = self.decoder(
            y, encoder_output, source_mask=source_mask, target_mask=target_mask
        )
        # 输出模块
        generator_output = self.generator(decoder_output)
        # 返回数据
        return generator_output


if __name__ == "__main__":
    # 1. 设定超参数
    src_vocab_size = 1000  # 假设源语言词典大小
    target_vocab_size = 1200  # 假设目标语言词典大小（可以不同）
    batch_size = 2
    src_len = 10
    tgt_len = 8
    d_model = 512

    # 2. 模拟随机输入张量 (词索引)
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    target = torch.randint(0, target_vocab_size, (batch_size, tgt_len))

    # 3. 实例化完整的 Transformer 模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
        d_model=d_model,
        n_head=8,
        n_layer=6,
        d_ff=2048,
        dropout_p=0.1,
    )

    # 4. 前向传播
    output = model(src, target)

    # 5. 检查输出
    print("最终输出维度:", output.shape)
    # 预期输出: torch.Size([2, 8, 1200])  -> [batch_size, target_seq_len, target_vocab_size]
    print("模型结构:")
    print(model)
