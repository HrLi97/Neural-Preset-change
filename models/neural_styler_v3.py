import torch, os, sys
import torch.nn as nn
sys.path.insert(1,os.path.abspath('..'))
from models.efficientnet import EfficientNet

class neural_styler(nn.Module):
    def __init__(self, cfg):
        super(neural_styler, self).__init__()
        self.cfg = cfg
        self.k = cfg.model.k
        self.style_encoder = EfficientNet.from_name(f'{cfg.model.style_encoder}', num_classes=(self.k ** 2)*2)
        self.transform_p = nn.Parameter(torch.rand(3, self.k))
        self.transform_q = nn.Parameter(torch.rand(self.k, 3))
        self.bias_p_z = nn.Parameter(torch.zeros(3, self.k))
        self.bias_q_z = nn.Parameter(torch.zeros(self.k, 1))
        self.bias_p_y = nn.Parameter(torch.zeros(3, self.k))
        self.bias_q_y = nn.Parameter(torch.zeros(self.k, 1))

    def get_r_and_d(self, style):
        """
        Args:
            style: style image (B x 3 x 256 x 256)
        """
        tmp_matrix = self.style_encoder(style)  # B x (k^2)*2
        
        # split tmp_matrix into r and d (B x k x k matrix each)
        r = tmp_matrix[:, :self.k**2].reshape(-1, self.k, self.k) # 16 * 16
        d = tmp_matrix[:, self.k**2:].reshape(-1, self.k, self.k)

        return r, d

    def forward(self, content, style, r_s=None, phase="full_train"):
        """
        Args:
            content: content image (B x 3 x H x W)
            style: style image (B x 3 x 256 x 256)
        """

        # batch-specific style whitening/coloring matrix (B x k x k matrix)
        if r_s is None:
            r_s, d_s = self.get_r_and_d(style)
        r_c, d_c = self.get_r_and_d(content)

        # calculate whitening transform matrix & coloring transform matrix
        transform_z = torch.matmul(torch.matmul(self.transform_p, d_c), self.transform_q)   # B x 3 x 3
        bias_z = torch.matmul(torch.matmul(self.bias_p_z, d_c), self.bias_q_z)               # B x 3 x 1
        z = torch.bmm(transform_z, content.reshape(-1, 3, content.shape[2]*content.shape[3])) + bias_z  # B x 3 x (H*W)
        z = z.reshape(-1, 3, content.shape[2], content.shape[3])  # 解耦特征（色彩归一化后的内容图）

        if phase == "pretrain":
            return z

        # 全量训练/推理阶段：计算风格迁移结果
        transform_y = torch.matmul(torch.matmul(self.transform_p, r_s), self.transform_q)   # B x 3 x 3
        bias_y = torch.matmul(torch.matmul(self.bias_p_y, r_s), self.bias_q_y)               # B x 3 x 1
        colored_content = (torch.bmm(transform_y, z.reshape(-1, 3, z.shape[2]*z.shape[3])) + bias_y).reshape(-1, 3, z.shape[2], z.shape[3])  # B x 3 x H x W
        
        # z代表色彩归一化后的内容图；colored_content代表最终风格迁移结果
        return z, colored_content
    