import torch
import torch.nn as nn

from losses.loss_fullySupervised import Focal_Loss, SoftIoULoss, SoftLoULoss1, muti_bce_loss_fusion, muti_SoftLoULoss1_fusion
from losses.loss_OHEM import MyWeightTopKLoss_Absolutly
from losses.loss_BCETopKLoss import MyWeightBCETopKLoss



def loss_chose(args):
    MyWgt = torch.Tensor(args.MyWgt)

    if args.loss_func == 'fullySup':
        cirterion = SoftIoULoss()
    elif args.loss_func == 'fullySup1':
        cirterion = SoftLoULoss1()
    elif args.loss_func == 'fullySup2':
        cirterion = muti_bce_loss_fusion()
    elif args.loss_func == 'fullySupBCE':
        cirterion = nn.BCEWithLogitsLoss(size_average=False)
    elif args.loss_func == 'FocalLoss':
        cirterion = Focal_Loss(alpha=MyWgt, gamma=2)
    elif args.loss_func == 'OHEM':
        cirterion = MyWeightTopKLoss_Absolutly(alpha=MyWgt, gamma=2, MaxClutterNum=args.MaxClutterNum,
                                     ProtectedArea=args.ProtectedArea)
    elif args.loss_func == 'HPM':
        cirterion = MyWeightBCETopKLoss(alpha=MyWgt, gamma=2, MaxClutterNum=args.MaxClutterNum,
                                     ProtectedArea=args.ProtectedArea)
    else:
        raise('An unexpected loss function!')

    return cirterion

