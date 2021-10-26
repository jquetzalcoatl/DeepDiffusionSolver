from torch import nn, abs


class RelativeErrorLoss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='none', eps=1e-10):
        super(RelativeErrorLoss, self).__init__()

        self.sa = size_average
        self.reduce = reduce
        self.reduction = reduction

        self.eps = eps

        self.mae = nn.L1Loss(size_average=self.sa, reduce=self.reduce, reduction=self.reduction)

    def forward(self, x, y):
        return self.mae(x, y) / (abs(x) + self.eps)
