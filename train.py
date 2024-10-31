import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from numpy import *
import numpy as np
import scipy.io as scio
import time
import os
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import auc

from MIRSDTDataLoader import TrainSetLoader, TestSetLoader
from IRDSTDataLoader import IRDST_TrainSetLoader, IRDST_TestSetLoader

from models.model_ISNet.train_ISNet import Get_gradientmask_nopadding, Get_gradient_nopadding

from models.model_config import model_chose, run_model
from losses.loss_config import loss_chose

from ShootingRules import ShootingRules



# def str2bool(v):
#     return v.lower() in ('yes', 'true', 't', '1')


def generate_savepath(args, epoch, epoch_loss):

    timestamp = time.time()
    CurTime = time.strftime("%Y_%m_%d__%H_%M", time.localtime(timestamp))

    SavePath = args.saveDir + args.model + '_SpatialDeepSup' + str(args.SpatialDeepSup) + '_' + args.loss_func + '/'
    ModelPath = SavePath + 'net_' + str(epoch+1) + '_epoch_' + str(epoch_loss) + '_loss_' + CurTime + '.pth'
    ParameterPath = SavePath + 'net_para_' + CurTime + '.pth'

    if not os.path.exists(args.saveDir):
        os.mkdir(args.saveDir)
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)

    return ModelPath, ParameterPath, SavePath


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Infrared_target_detection_overall')
    parser.add_argument('--DataPath',  type=str, default='./dataset/', help='Dataset path [default: ./dataset/]')
    parser.add_argument('--dataset',   type=str, default='NUDT-MIRSDT', help='Dataset name [dafult: NUDT-MIRSDT]')
    parser.add_argument('--align',  default='False', action='store_true', help='align input frames')
    parser.add_argument('--training_rate', type=int, default=1, help='Rate of samples in training (1/n) [default: 1]')
    parser.add_argument('--saveDir',   type=str, default='./results/',
                            help='Save path [defaule: ./results/]')
    parser.add_argument('--train',    type=int, default=0)
    parser.add_argument('--test',     type=int, default=1)
    parser.add_argument('--pth_path', type=str, default='./results/ResUNet_DTUM_SpatialDeepSupFalse_fullySup/ResUNet_DTUM.pth', help='Trained model path')

    # train
    parser.add_argument('--model',     type=str, default='ResUNet_DTUM',
                        help='ResUNet_DTUM, DNANet_DTUM, ACM, ALCNet, ResUNet, DNANet, ISNet, UIU')
    parser.add_argument('--loss_func', type=str, default='fullySup',
                        help='HPM, FocalLoss, OHEM, fullySup, fullySup1(ISNet), fullySup2(UIU)')
    parser.add_argument('--fullySupervised', default=True)
    parser.add_argument('--SpatialDeepSup',  default=False)
    parser.add_argument('--batchsize', type=int,   default=1)
    parser.add_argument('--epochs',    type=int,   default=20)
    parser.add_argument('--lrate',     type=float, default=0.001)
    # parser.add_argument('--lrate_min', type=float, default=1e-5)

    # loss
    parser.add_argument('--MyWgt',     default=[0.1667, 0.8333], help='Weights of positive and negative samples')
    parser.add_argument('--MaxClutterNum', type=int, default=39, help='Clutter samples in loss [default: 39]')
    parser.add_argument('--ProtectedArea', type=int, default=2,  help='1,2,3...')

    # GPU
    parser.add_argument('--DataParallel',     default=False,    help='Use one gpu or more')
    parser.add_argument('--device', type=str, default="cuda:0", help='use comma for multiple gpus')

    args = parser.parse_args()

    # the parser
    return args




class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # model
        self.net = model_chose(args.model, args.loss_func, args.SpatialDeepSup)
        if args.DataParallel:
            self.net = nn.DataParallel(self.net)  #, device_ids=[0,1,2]).cuda()
        self.net = self.net.to(self.device)

        train_path = args.DataPath + args.dataset + '/'
        self.test_path = train_path
        if args.dataset == 'NUDT-MIRSDT':
            self.train_dataset = TrainSetLoader(train_path, fullSupervision=args.fullySupervised)
            self.val_dataset = TestSetLoader(self.test_path)
        elif args.dataset == 'IRDST':
            self.train_dataset = IRDST_TrainSetLoader(train_path, fullSupervision=args.fullySupervised, align=args.align)
            self.val_dataset = IRDST_TestSetLoader(self.test_path, align=args.align)
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, )

        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lrate, betas=(0.9, 0.99))
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5, last_epoch=-1)

        self.criterion = loss_chose(args)
        self.criterion2 = nn.BCELoss()
        self.eval_metrics = ShootingRules()

        self.loss_list = []
        self.Gain = 100
        self.epoch_loss = 0

        ########### save ############
        self.ModelPath, self.ParameterPath, self.SavePath = generate_savepath(args, 0, 0)
        self.test_save = self.SavePath[0:-1] + '_visualization/'
        self.writeflag = 1
        self.save_flag = 1
        if self.save_flag == 1 and not os.path.exists(self.test_save):
            os.mkdir(self.test_save)


    def training(self, epoch):
        args = self.args
        running_loss = 0.0
        loss_last = 0.0
        self.net.train()
        for i, data in enumerate(tqdm(self.train_loader), 0):
            if i % args.training_rate != 0:
                continue

            SeqData_t, TgtData_t, m, n = data
            SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)  # b,t,m,n  // b,1,m.n
            self.optimizer.zero_grad()

            outputs = run_model(self.net, args.model, SeqData, 0, 0)
            if isinstance(outputs, list):
                if isinstance(outputs[0], tuple):
                    outputs[0] = outputs[0][0]
            elif isinstance(outputs, tuple):
                outputs = outputs[0]

            if 'DNANet' in args.model:
                loss = 0
                if isinstance(outputs, list):
                    for output in outputs:
                        loss += self.criterion(output, TgtData.float())
                    loss /= len(outputs)
                else:
                    loss = self.criterion(outputs, TgtData.float())
            elif 'ISNet' in args.model and args.loss_func == 'fullySup1':   ## and 'ISNet_woTFD' not in args.model
                edge = torch.cat([TgtData, TgtData, TgtData], dim=1).float()  # b, 3, m, n
                gradmask = Get_gradientmask_nopadding()
                edge_gt = gradmask(edge)
                loss_io = self.criterion(outputs[0], TgtData.float())
                if args.fullySupervised:
                    outputs[1] = torch.sigmoid(outputs[1])
                    loss_edge = 10 * self.criterion2(outputs[1], edge_gt) + self.criterion(outputs[1], edge_gt)
                else:
                    loss_edge = 10 * self.criterion2(torch.sigmoid(outputs[1]), edge_gt) + self.criterion(outputs[1], edge_gt.float())
                if 'DTUM' in args.model or not args.fullySupervised:
                    alpha = 0.1
                else:
                    alpha = 1
                loss = loss_io + alpha * loss_edge
            elif 'UIU' in args.model:
                if 'fullySup2' in args.loss_func:
                    loss0, loss = self.criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], TgtData.float())
                    if not args.SpatialDeepSup:
                        loss = loss0   ## without SDS
                else:
                    loss = 0
                    if not args.SpatialDeepSup:
                        loss = self.criterion(outputs[0], TgtData.float())
                    else:
                        for output in outputs:
                            loss += self.criterion(output, TgtData.float())
            else:
                loss = self.criterion(outputs, TgtData.float())

            '''
            LogSoftmax = nn.Softmax(dim=1)
            outputs=torch.squeeze(outputs, 2)
            Outputs_Max = LogSoftmax(outputs)
            fig=plt.figure()
            ShowInd=0
            plt.subplot(221); plt.imshow(SeqData.data.cpu().numpy()[ShowInd,0,4,:,:], cmap='gray')
            plt.subplot(222); plt.imshow(TgtData.data.cpu().numpy()[ShowInd,0,:,:], cmap='gray')
            plt.subplot(223); plt.imshow(outputs.data.cpu().numpy()[ShowInd,1,:,:], cmap='gray')
            plt.subplot(224); plt.imshow(Outputs_Max.data.cpu().numpy()[ShowInd,1,:,:], cmap='gray')
            plt.show()
            '''

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if epoch == 0 and (i + 1) % 50 == 0:
                loss_50 = running_loss - loss_last
                loss_last = running_loss
                print('model: %s, epoch=%d, i=%d, loss.item=%.10f' % (args.model + args.loss_func, epoch, i, loss_50))

        self.epoch_loss = running_loss / i * self.Gain
        print('model: %s, epoch: %d, loss: %.10f' % (args.model + args.loss_func, epoch + 1, self.epoch_loss))
        ########################################
        self.scheduler.step()
        # if optimizer.state_dict()['param_groups'][0]['lr'] < args.lrate_min:
        #     optimizer.state_dict()['param_groups'][0]['lr'] = args.lrate_min

        self.loss_list.append(self.epoch_loss)


    def validation(self, epoch):
        args = self.args
        txt = np.loadtxt(self.test_path + 'test.txt', dtype=bytes).astype(str)
        self.net.eval()

        low_snr3 = [47,56,59,76,92,101,105,119]
        high_snr3 = [85,86,87,88,89,90,91,93,94,95,96,97]

        Th_Seg = np.array(
            [0, 1e-30, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
             1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, .15, 0.2, .25, 0.3, .35, 0.4, .45, 0.5, .55, 0.6, .65, 0.7, .75,
             0.8, .85, 0.9, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999, 0.999999, 0.9999999, 1])
        if epoch < args.epochs-1:
            Th_Seg = np.array([0, 1e-1, 0.2, 0.3, .35, 0.4, .45, 0.5, .55, 0.6, .65, 0.7, 0.8, 0.9, 0.95, 1])
        FalseNumAll = np.zeros([20,len(Th_Seg)])
        TrueNumAll = np.zeros([20,len(Th_Seg)])
        TgtNumAll = np.zeros([20,len(Th_Seg)])

        OldFlag = 0
        Old_Feat = torch.zeros([1,32,4,512,512]).to(self.device)  # interface for iteration input
        pixelsNumber = np.zeros(20)
        time_start = time.time()
        for i, data in enumerate(tqdm(self.val_loader), 0):
            # if i > 5: break
            if i % 100 == 0:
                OldFlag = 0
            else:
                OldFlag = 1
            Seq_num = int(txt[i].split('Sequence')[1].split('/Mix')[0])
            index = (low_snr3+high_snr3).index(Seq_num)

            with torch.no_grad():
                SeqData_t, TgtData_t, m, n = data
                SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)

                outputs = run_model(self.net, args.model, SeqData, Old_Feat, OldFlag)
                if 'ISNet' in args.model:   ## and args.model != 'ISNet_woTFD'
                    edge_out = torch.sigmoid(outputs[1]).data.cpu().numpy()[0, 0, 0:m, 0:n]

                if isinstance(outputs, list):
                    outputs = outputs[0]
                if isinstance(outputs, tuple):
                    Old_Feat = outputs[1]
                    outputs = outputs[0]
                outputs = torch.squeeze(outputs, 2)

                Outputs_Max = torch.sigmoid(outputs)
                TestOut = Outputs_Max.data.cpu().numpy()[0, 0, 0:m, 0:n]

                pixelsNumber[index] += m * n
                if self.save_flag:
                    img = Image.fromarray(uint8(TestOut * 255))
                    folder_name = "%sSequence%d/" % (self.test_save, Seq_num)
                    if not os.path.exists(folder_name):
                        os.mkdir(folder_name)
                    name = folder_name + ('%05d.png' % (i % 100 + 1))
                    img.save(name)
                    save_name = folder_name + ('%05d.mat' % (i % 100 + 1))
                    scio.savemat(save_name, {'TestOut': TestOut})

                    if 'ISNet' in args.model:   ## and args.model != 'ISNet_woTFD'
                        edge_out = Image.fromarray(uint8(edge_out * 255))
                        edge_name = folder_name + ('%05d_EdgeOut.png' % (i % 100 + 1))
                        edge_out.save(edge_name)

                # the statistics for detection result
                if self.writeflag:
                    for th_i in range(len(Th_Seg)):
                        FalseNum, TrueNum, TgtNum = self.eval_metrics(Outputs_Max[:,:,:m,:n], TgtData[:,:,:m,:n], Th_Seg[th_i])

                        FalseNumAll[index, th_i] = FalseNumAll[index, th_i] + FalseNum
                        TrueNumAll[index, th_i] = TrueNumAll[index, th_i] + TrueNum
                        TgtNumAll[index, th_i] = TgtNumAll[index, th_i] + TgtNum

        time_end = time.time()
        print('FPS=%.3f' % ((i+1)/(time_end-time_start)))

        if self.writeflag:

            Pd_lSNR = np.sum(TrueNumAll[0:8, :], axis=0) / np.sum(TgtNumAll[0:8, :], axis=0)
            Pd_hSNR = np.sum(TrueNumAll[8:, :], axis=0) / np.sum(TgtNumAll[8:, :], axis=0)
            Pd_all = np.sum(TrueNumAll[:, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)
            Fa_lSNR = np.sum(FalseNumAll[0:8, :], axis=0) / pixelsNumber[0:8].sum()
            Fa_hSNR = np.sum(FalseNumAll[8:, :], axis=0) / pixelsNumber[8:].sum()
            Fa_all = np.sum(FalseNumAll[:, :], axis=0) / pixelsNumber.sum()
            auc_lSNR = auc(Fa_lSNR, Pd_lSNR)
            auc_hSNR = auc(Fa_hSNR, Pd_hSNR)
            auc_all = auc(Fa_all, Pd_all)

            writelines = open(self.SavePath + 'Epoch' + str(epoch+1) + '_ROC_ShootingRules.txt', 'w')
            for i in range(20):
                seq = (low_snr3+high_snr3)[i]
                writelines.write('Seq' + str(seq) + 'results:\n')
                for seg_i in range(len(Th_Seg)):
                    writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[seg_i], TrueNumAll[i, seg_i], TgtNumAll[i, seg_i],
                                TrueNumAll[i, seg_i] / TgtNumAll[i, seg_i], FalseNumAll[i, seg_i], FalseNumAll[i, seg_i] / pixelsNumber[i]))

            writelines.write('Low SNR results:\tAUC:%.5f\n' % auc_lSNR)
            for th_i in range(len(Th_Seg)):
                writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[0:8, th_i].sum(),
                                TgtNumAll[0:8, th_i].sum(), TrueNumAll[0:8, th_i].sum() / TgtNumAll[0:8, th_i].sum(),
                                FalseNumAll[0:8, th_i].sum(), FalseNumAll[0:8, th_i].sum() / pixelsNumber[0:8].sum()))

            writelines.write('High SNR results:\tAUC:%.5f\n' % auc_hSNR)
            for th_i in range(len(Th_Seg)):
                writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[8:, th_i].sum(),
                                TgtNumAll[8:, th_i].sum(), TrueNumAll[8:, th_i].sum() / TgtNumAll[8:, th_i].sum(),
                                FalseNumAll[8:, th_i].sum(), FalseNumAll[8:, th_i].sum() / pixelsNumber[8:].sum()))

            writelines.write('Final results:\tAUC:%.5f\n' % auc_all)
            for th_i in range(len(Th_Seg)):
                writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[:, th_i].sum(),
                                TgtNumAll[:, th_i].sum(), TrueNumAll[:, th_i].sum() / TgtNumAll[:, th_i].sum(),
                                FalseNumAll[:, th_i].sum(), FalseNumAll[:, th_i].sum() / pixelsNumber.sum()))
            writelines.close()

            seg = 29
            if epoch < args.epochs-1:
                seg = 7
            print('model: %s, epoch: %d, Th_Seg = %.4e, PD:[%d, %.5f], FA:[%d, %.4e], AUC:%.5f' % (args.model + args.loss_func, epoch + 1,
                Th_Seg[seg], TrueNumAll[:,seg].sum(), Pd_all[seg], FalseNumAll[:,seg].sum(), Fa_all[seg], auc_all))


    def savemodel(self, epoch):
        self.ModelPath, self.ParameterPath, self.SavePath = generate_savepath(self.args, epoch, self.epoch_loss)
        torch.save(self.net, self.ModelPath)
        torch.save(self.net.state_dict(), self.ParameterPath)
        print('save net OK in %s' % self.ModelPath)


    def saveloss(self):
        CurTime = time.strftime("%Y_%m_%d__%H_%M", time.localtime())
        print(CurTime)

        ###########save lost_list
        LossMatSavePath = self.SavePath + 'loss_list_' + CurTime + '.mat'
        scio.savemat(LossMatSavePath, mdict={'loss_list': self.loss_list})

        ############plot
        x1 = range(self.args.epochs)
        y1 = self.loss_list
        fig = plt.figure()
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        LossJPGSavePath = self.SavePath + 'train_loss_' + CurTime + '.jpg'
        plt.savefig(LossJPGSavePath)
        # plt.show()
        print('finished Show!')




if __name__ == '__main__':
    args = parse_args()
    StartTime = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    print(StartTime)

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # torch.cuda.set_device(0)

    trainer = Trainer(args)
    if args.train == 1:
        for epoch in range(args.epochs):
            trainer.training(epoch)

            if (epoch+1)%10 == 0:
                trainer.savemodel(epoch)
                trainer.validation(epoch)
        # trainer.savemodel()
        trainer.saveloss()
        print('finished training!')
    if args.test == 1:
        #####################################################
        trainer.ModelPath = args.pth_path
        trainer.test_save = trainer.SavePath[0:-1] + '_visualization/'
        trainer.net = torch.load(trainer.ModelPath, map_location=trainer.device)
        print('load OK!')
        epoch = args.epochs
        #####################################################
        trainer.validation(epoch)







