import torch.nn.functional as F


class LabelSmoothingCrossEntropy(object):

    def __init__(self, eps=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps

    def forward(self, x, targets):
        log_probs = F.log_softmax(x, dim=1) # 实现log(p)   N * K

        H_qp = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))  # 计算各个样本的label位置的交叉熵损失 N * 1
        H_qp = H_qp.squeeze(1)

        # H(u, p) 由于u是均匀分布，等价于求均值
        H_up = -log_probs.mean()

        loss = (1 - self.eps) * H_qp + self.eps * H_up

        return loss.mean()



