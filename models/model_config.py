import torch

from models.model_DNANet import DNANet, DNANet_DTUM
from models.model_res_UNet import res_UNet, res_UNet_DTUM
from models.model_ACM import ACM
from models.model_alcnet import ASKCResNetFPN as ALCNet
from models.model_alcnet import ALCNet_DTUM
from models.model_DNANet import Res_CBAM_block
from models.model_res_UNet import Res_block
from models.model_ISNet.ISNet import ISNet, ISNet_woTFD, ISNet_DTUM
from models.model_ISNet.train_ISNet import Get_gradient_nopadding
from models.model_UIU.uiunet import UIUNET, UIUNET_DTUM

# from thop import profile
# from thop import clever_format


def model_chose(model, loss_func, SpatialDeepSup):
    num_classes = 1

    if model == 'ACM':
        net = ACM(in_channels=3, layers=[3]*3, fuse_mode='AsymBi', tiny=False, classes=num_classes)

    elif model == 'ALCNet':
        net = ALCNet(layers=[4]*4, channels=[8,16,32,64,128], shift=13, pyramid_mod='AsymBi', scale_mode='Single',
                     act_dilation=16, fuse_mode='AsymBi',pyramid_fuse='Single', r=2, classes=num_classes)
    elif model == 'ALCNet_DTUM':
        net = ALCNet_DTUM(layers=[4]*4, channels=[8,16,32,64,128], shift=13, pyramid_mod='AsymBi', scale_mode='Single',
                     act_dilation=16, fuse_mode='AsymBi',pyramid_fuse='Single', r=2, classes=num_classes)

    elif model == 'DNANet':
        net = DNANet(num_classes=num_classes, input_channels=3, block=Res_CBAM_block, num_blocks=[2,2,2,2], nb_filter=[16,32,64,128,256])
    elif model == 'DNANet_DTUM':
        net = DNANet_DTUM(num_classes=num_classes, input_channels=3, block=Res_CBAM_block, num_blocks=[2,2,2,2], nb_filter=[16,32,64,128,256], deep_supervision=SpatialDeepSup)

    elif model == 'ResUNet':
        net = res_UNet(num_classes=num_classes, input_channels=3, block=Res_block, num_blocks=[2,2,2,2], nb_filter=[8,16,32,64,128])
    elif model == 'ResUNet_DTUM':
        net = res_UNet_DTUM(num_classes=num_classes, input_channels=3, block=Res_block, num_blocks=[2,2,2,2], nb_filter=[8,16,32,64,128])

    elif model == 'ISNet':
        net = ISNet(layer_blocks=[4]*3, channels=[8,16,32,64], num_classes=num_classes)
    # elif model == 'ISNet_woTFD':
    #     net = ISNet_woTFD(layer_blocks=[4]*3, channels=[8,16,32,64], num_classes=num_classes)
    elif model == 'ISNet_DTUM':
        net = ISNet_DTUM(layer_blocks=[4] * 3, channels=[8, 16, 32, 64], num_classes=num_classes)

    elif model == 'UIU':
        net = UIUNET(in_ch=3, out_ch=num_classes)
    elif model == 'UIU_DTUM':
        net = UIUNET_DTUM(in_ch=3, out_ch=num_classes, deep_supervision=SpatialDeepSup)

    return net


def run_model(net, model, SeqData, Old_Feat, OldFlag):

    # Old_Feat = SeqData[:,:,:-1, :,:] * 0  # interface for iteration input
    # OldFlag = 1  # 1: i

    if model=='DNANet' or model=='ResUNet' or model=='ACM' or model=='ALCNet':   ## or model=='ISNet_woTFD'
        input = SeqData[:, :, -1, :, :].repeat(1, 3, 1, 1)
        outputs = net(input)
    elif model=='DNANet_DTUM' or model=='ResUNet_DTUM' or model=='ALCNet_DTUM':
        input = SeqData.repeat(1, 3, 1, 1, 1)
        outputs = net(input, Old_Feat, OldFlag)

    elif model == 'ISNet':
        input = SeqData[:, :, -1, :, :].repeat(1, 3, 1, 1)
        grad = Get_gradient_nopadding()
        edge_in = grad(input)
        outputs, edge_outs = net(input, edge_in)
        outputs = [outputs, edge_outs]
    elif model == 'ISNet_DTUM':
        input = SeqData.repeat(1, 3, 1, 1, 1)
        outputs, edge_outs = net(input, Old_Feat, OldFlag)
        outputs = [outputs, edge_outs]

    elif model == 'UIU':
        input = SeqData[:, :, -1, :, :].repeat(1, 3, 1, 1)
        d0, d1,d2,d3,d4,d5,d6 = net(input)
        outputs = [d0, d1, d2, d3, d4, d5, d6]
    elif model == 'UIU_DTUM':
        input = SeqData.repeat(1, 3, 1, 1, 1)
        d0, d1,d2,d3,d4,d5,d6 = net(input, Old_Feat, OldFlag)
        outputs = [d0, d1, d2, d3, d4, d5, d6]

    # if OldFlag == 0:
    #     if 'DTUM' in model:
    #         flops, params = profile(net, inputs=(input, Old_Feat, OldFlag))   # runtimeerror cpu : net.module
    #     elif model == 'ISNet':
    #         flops, params = profile(net, inputs=(input, edge_in))
    #     else:
    #         flops, params = profile(net, inputs=(input, ))
    #     flops, params = clever_format([flops, params], '%.3f')
    #     print(flops, params)

    return outputs
