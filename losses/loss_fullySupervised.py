import torch
import torch.nn as nn
import torch.nn.functional as F


class Focal_Loss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha(0), alpha(1)])
        # if isinstance(alpha, (float, int)): self.alpha = torch.Tensor(alpha)
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)


    def forward(self, input, target):
        if input.dim() > 4:
            input = torch.squeeze(input, 2)
        logpt = F.logsigmoid(input)
        logpt_bk = F.logsigmoid(-input)
        pt = logpt.data.exp()
        pt_bk = logpt_bk.data.exp()
        loss = -self.alpha[1]*(1-pt)**self.gamma*target*logpt - self.alpha[0]*pt_bk**self.gamma*(1-target)*logpt_bk

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class SoftIoULoss(nn.Module):
    def __init__(self,gamma=0, alpha=None, size_average=False):
        super(SoftIoULoss, self).__init__()
        self.gamma=gamma
        self.alpha=alpha
        self.size_average=size_average
        if isinstance(alpha, (float, int)): self.alpha=torch.Tensor([alpha(0), alpha(1)])
        # if isinstance(alpha, (float, int)): self.alpha = torch.Tensor(alpha)
        if isinstance(alpha, list): self.alpha=torch.Tensor(alpha)

    def forward(self, pred, target):
        if pred.dim() > 4:
            pred = torch.squeeze(pred, 2)

        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss



class SoftLoULoss1(nn.Module):
    def __init__(self, batch=32):
        super(SoftLoULoss1, self).__init__()
        self.batch = batch
        # self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        if pred.dim() > 4:
            pred = torch.squeeze(pred, 2)

        pred = torch.sigmoid(pred)
        smooth = 0.00

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        # loss1 = self.bce_loss(pred, target)
        return loss




class muti_SoftLoULoss1_fusion(nn.Module):
    def __init__(self, size_average=True):
        super(muti_SoftLoULoss1_fusion, self).__init__()

        self.softIou = SoftLoULoss1()

    def forward(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        if d0.dim() > 4:
            d0 = torch.squeeze(d0, 2)

        loss0 = self.softIou(d0, labels_v)
        loss1 = self.softIou(d1, labels_v)
        loss2 = self.softIou(d2, labels_v)
        loss3 = self.softIou(d3, labels_v)
        loss4 = self.softIou(d4, labels_v)
        loss5 = self.softIou(d5, labels_v)
        loss6 = self.softIou(d6, labels_v)

        loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6) / 7

        return loss0, loss




class muti_bce_loss_fusion(nn.Module):
    def __init__(self, size_average=True):
        super(muti_bce_loss_fusion, self).__init__()

        self.bce_loss = nn.BCELoss(size_average=size_average)

    def forward(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        if d0.dim() > 4:
            d0 = torch.squeeze(d0, 2)

        loss0 = self.bce_loss(torch.sigmoid(d0), labels_v)
        loss1 = self.bce_loss(torch.sigmoid(d1), labels_v)
        loss2 = self.bce_loss(torch.sigmoid(d2), labels_v)
        loss3 = self.bce_loss(torch.sigmoid(d3), labels_v)
        loss4 = self.bce_loss(torch.sigmoid(d4), labels_v)
        loss5 = self.bce_loss(torch.sigmoid(d5), labels_v)
        loss6 = self.bce_loss(torch.sigmoid(d6), labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        return loss0, loss




# class bce_loss_NPos(nn.Module):
#     def __init__(self):
#         super(bce_loss_NPos, self).__init__()
#
#         self.bce_loss = nn.BCELoss(reduce=False)
#
#     def forward(self, input, target):
#         input = torch.sigmoid(input)
#         losses = self.bce_loss(input, target)
#         positives = losses[target==1]
#
#         return loss





class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


