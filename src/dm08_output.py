from torch import nn


class Generator(nn.Module):
    def __init__(self, vocab_size,d_model):
        """

        :param d_model: 隐藏层维度
        :param vocab_size: 词典大小
        """
        super().__init__()
        # 全连接线性层
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.project(x)
        return x
