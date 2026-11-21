from torch import nn

import dm01_input
import dm02_encoder_element


# 子层连接结构
class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout_p=0.1):
        """
        参数初始化
        :param d_model: 隐藏层维度
        :param dropout_p: 随机失活概率
        """
        super().__init__()
        # 定义规范化层
        self.norm = dm02_encoder_element.LayerNorm(d_model)
        # 随机失活层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, sublayer_out):
        """
        子层连接
        :param x: 输入子层的张量
        :param sublayer_out: 子层的输出
        :return:
        """
        # 残差连接 + 层规范化
        x = self.norm(x + self.dropout(sublayer_out))
        return x


def dm_test_sublayer_connection():
    pass


if __name__ == "__main__":
    pass
