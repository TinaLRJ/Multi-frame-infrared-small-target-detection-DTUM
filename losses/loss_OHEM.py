import torch
# from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class MyWeightTopKLoss_Absolutly(nn.Module):
    def __init__(self,gamma=0, alpha=None, size_average=False, MaxClutterNum=39, ProtectedArea=2):
        super(MyWeightTopKLoss_Absolutly, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduce=False)

        self.MaxClutterNum = MaxClutterNum
        self.ProtectedArea = ProtectedArea
        self.gamma=gamma
        self.alpha=alpha
        self.size_average = size_average
        if isinstance(alpha, (float, int)): self.alpha=torch.Tensor([alpha(0), alpha(1)])
        if isinstance(alpha, list): self.alpha=torch.Tensor(alpha)


    def forward(self, input, target):

        if input.dim() > 4:
            input = torch.squeeze(input, 2)

        ## target surrounding = 2
        template = torch.ones(1, 1, 2*self.ProtectedArea+1, 2*self.ProtectedArea+1).to(input.device)
        target_prot = F.conv2d(target.float(), template, stride=1, padding=self.ProtectedArea)
        target_prot = (target_prot > 0).float()

        with torch.no_grad():
            loss_wise = self.bce_loss(input, target)
            loss_p = loss_wise * (1 - target_prot)
            batch_l = loss_p.shape[0]
            Wgt = torch.zeros(batch_l, 1, 512, 512)
            for ls in range(batch_l):
                loss_ls = loss_p[ls, :, :, :].reshape(-1)
                loss_topk, indices = torch.topk(loss_ls, self.MaxClutterNum)
                for i in range(self.MaxClutterNum):
                    Wgt[ls, 0, indices[i] // 512, indices[i] % 512] = 1

            WgtData_New = Wgt.to(input.device) + target.float()
            WgtData_New[WgtData_New > 1] = 1

        logpt = F.logsigmoid(input)
        logpt_bk = F.logsigmoid(-input)
        pt = logpt.data.exp()
        pt_bk = 1 - logpt_bk.data.exp()
        loss = -self.alpha[1]*(1-pt)**self.gamma*target*logpt - self.alpha[0]*pt_bk**self.gamma*(1-target)*logpt_bk

        loss = loss * WgtData_New

        return loss.sum()

        # if input.dim()>2:
        #     input=input.view(input.size(0), input.size(1),-1)   # N,C,D,H,W=>N,C,D*H*W
        #     input=input.transpose(1,2)                          # N,C,D*H*W=>N, D*H*W, C
        #     input=input.contiguous().view(-1,input.size(2))     # N,D*H*W,C=>N*D*H*W, C
        #
        #     WgtData_New = WgtData_New.view(WgtData_New.size(0), WgtData_New.size(1), -1)    # N,C,D,H,W=>N,C,D*H*W
        #     WgtData_New = WgtData_New.transpose(1, 2)                               # N,C,D*H*W=>N, D*H*W,C
        #     WgtData_New = WgtData_New.contiguous().view(-1, WgtData_New.size(2))        # N,D*H*W,C=>N*D*H*W,C
        #
        # target = target.view(-1,1)           #N,D,H,W=>1,N*D*H*W
        # logpt = F.log_softmax(input)
        # logpt = logpt.gather(1,target)
        # logpt=logpt*WgtData_New             #weight
        # logpt=logpt.view(-1)
        # pt=logpt.data.exp()
        #
        # if self.alpha is not None:
        #     if self.alpha.type()!=input.data.type():
        #         self.alpha=self.alpha.type_as(input.data).to(input.device)
        #     at=self.alpha.gather(0,target.data.view(-1))
        #     logpt=logpt*at
        #
        # loss=-1*(1-pt)**self.gamma*logpt
        #
        # return loss.sum()


