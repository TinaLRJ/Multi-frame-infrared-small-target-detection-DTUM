import numpy as np
from sklearn.metrics import auc


def writeNUDTMIRSDT_ROC(FalseNumBatch, TrueNumBatch, TgtNumBatch, pixelsNumBatch, Th_Seg, test_txt, save_path, args, epoch):
    low_snr3 = [47, 56, 59, 76, 92, 101, 105, 119]
    high_snr3 = [85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97]
    seq_names = list(dict.fromkeys([int(x.split('/')[0].split('Sequence')[1]) for x in test_txt]))
    index = [seq_names.index(i) for i in low_snr3+high_snr3]

    FalseNumAll = np.array(FalseNumBatch).reshape((20, -1, len(Th_Seg))).sum(axis=1)
    TrueNumAll = np.array(TrueNumBatch).reshape((20, -1, len(Th_Seg))).sum(axis=1)
    TgtNumAll = np.array(TgtNumBatch).reshape((20, -1, len(Th_Seg))).sum(axis=1)
    pixelsNumber = np.array(pixelsNumBatch).reshape(20, -1).sum(axis=1)

    Pd_lSNR = np.sum(TrueNumAll[index[0:8], :], axis=0) / np.sum(TgtNumAll[index[0:8], :], axis=0)
    Pd_hSNR = np.sum(TrueNumAll[index[8:], :], axis=0) / np.sum(TgtNumAll[index[8:], :], axis=0)
    Pd_all = np.sum(TrueNumAll[:, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)
    Fa_lSNR = np.sum(FalseNumAll[index[0:8], :], axis=0) / pixelsNumber[index[0:8]].sum()
    Fa_hSNR = np.sum(FalseNumAll[index[8:], :], axis=0) / pixelsNumber[index[8:]].sum()
    Fa_all = np.sum(FalseNumAll[:, :], axis=0) / pixelsNumber.sum()
    auc_lSNR = auc(Fa_lSNR, Pd_lSNR)
    auc_hSNR = auc(Fa_hSNR, Pd_hSNR)
    auc_all = auc(Fa_all, Pd_all)

    writelines = open(save_path + 'Epoch' + str(epoch+1) + '_ROC_ShootingRules.txt', 'w')
    for i in range(20):
        seq = (low_snr3 + high_snr3)[i]
        writelines.write('Seq' + str(seq) + 'results:\n')
        for seg_i in range(len(Th_Seg)):
            writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (
                Th_Seg[seg_i], TrueNumAll[index[i], seg_i], TgtNumAll[index[i], seg_i],
                TrueNumAll[index[i], seg_i] / TgtNumAll[index[i], seg_i], FalseNumAll[index[i], seg_i],
                FalseNumAll[index[i], seg_i] / pixelsNumber[index[i]]))

    writelines.write('Low SNR results:\tAUC:%.5f\n' % auc_lSNR)
    for th_i in range(len(Th_Seg)):
        writelines.write(
            'Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[index[0:8], th_i].sum(),
                                                               TgtNumAll[index[0:8], th_i].sum(),
                                                               TrueNumAll[index[0:8], th_i].sum() / TgtNumAll[index[0:8],
                                                                                             th_i].sum(),
                                                               FalseNumAll[index[0:8], th_i].sum(),
                                                               FalseNumAll[index[0:8], th_i].sum() / pixelsNumber[
                                                                                              index[0:8]].sum()))

    writelines.write('High SNR results:\tAUC:%.5f\n' % auc_hSNR)
    for th_i in range(len(Th_Seg)):
        writelines.write(
            'Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[index[8:], th_i].sum(),
                                                               TgtNumAll[index[8:], th_i].sum(),
                                                               TrueNumAll[index[8:], th_i].sum() / TgtNumAll[index[8:],
                                                                                            th_i].sum(),
                                                               FalseNumAll[index[8:], th_i].sum(),
                                                               FalseNumAll[index[8:], th_i].sum() / pixelsNumber[
                                                                                             index[8:]].sum()))

    writelines.write('Final results:\tAUC:%.5f\n' % auc_all)
    for th_i in range(len(Th_Seg)):
        writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[:, th_i].sum(),
                                                                            TgtNumAll[:, th_i].sum(),
                                                                            TrueNumAll[:, th_i].sum() / TgtNumAll[:,
                                                                                                        th_i].sum(),
                                                                            FalseNumAll[:, th_i].sum(),
                                                                            FalseNumAll[:,
                                                                            th_i].sum() / pixelsNumber.sum()))
    writelines.close()

    seg = list(Th_Seg).index(0.5)
    print('model: %s, epoch: %d, Th_Seg = %.4e, PD:[%d, %.5f], FA:[%d, %.4e], AUC:%.5f' % (
        args.model + args.loss_func, epoch + 1, Th_Seg[seg], TrueNumAll[:, seg].sum(),
        Pd_all[seg], FalseNumAll[:, seg].sum(), Fa_all[seg], auc_all))
    return


def writeIRSeq_ROC(FalseNumBatch, TrueNumBatch, TgtNumBatch, pixelsNumBatch, Th_Seg, save_path, args, epoch):
    FalseNumAll = np.array(FalseNumBatch).sum(axis=0)
    TrueNumAll = np.array(TrueNumBatch).sum(axis=0)
    TgtNumAll = np.array(TgtNumBatch).sum(axis=0)
    pixelsNumber = np.array(pixelsNumBatch).sum(axis=0)

    Pd_all = TrueNumAll / TgtNumAll
    Fa_all = FalseNumAll / pixelsNumber
    auc_all = auc(Fa_all, Pd_all)

    writelines = open(save_path + 'Epoch' + str(epoch+1) + '_ROC_ShootingRules.txt', 'w')

    writelines.write('Final results:\tAUC:%.5f\n' % auc_all)
    for th_i in range(len(Th_Seg)):
        writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[th_i],
                                                        TgtNumAll[th_i], TrueNumAll[th_i] / TgtNumAll[th_i],
                                                        FalseNumAll[th_i], FalseNumAll[th_i] / pixelsNumber))
    writelines.close()

    seg = list(Th_Seg).index(0.5)
    print('model: %s, epoch: %d, Th_Seg = %.4e, PD:[%d, %.5f], FA:[%d, %.4e], AUC:%.5f' % (
        args.model + args.loss_func, epoch + 1, Th_Seg[seg], TrueNumAll[:, seg].sum(),
        Pd_all[seg], FalseNumAll[:, seg].sum(), Fa_all[seg], auc_all))
    return



def writeCSIG_ROC(FalseNumBatch, TrueNumBatch, TgtNumBatch, pixelsNumBatch, Th_Seg, save_path, args, epoch):
    FalseNumAll = np.array(FalseNumBatch).reshape((-1, len(Th_Seg))).sum(axis=0)
    TrueNumAll = np.array(TrueNumBatch).reshape((-1, len(Th_Seg))).sum(axis=0)
    TgtNumAll = np.array(TgtNumBatch).reshape((-1, len(Th_Seg))).sum(axis=0)
    pixelsNumBatch = np.sum([x.item() if torch.is_tensor(x) else x for x in pixelsNumBatch])

    Pd_all = TrueNumAll / TgtNumAll
    Fa_all = FalseNumAll / pixelsNumBatch
    auc_all = auc(Fa_all, Pd_all)

    writelines = open(save_path + 'Epoch' + str(epoch+1) + '_ROC_ShootingRules.txt', 'w')

    writelines.write('Final results:\tAUC:%.5f\n' % auc_all)
    for th_i in range(len(Th_Seg)):
        writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[th_i],
                                                                            TgtNumAll[th_i],
                                                                            TrueNumAll[th_i] / TgtNumAll[
                                                                                                        th_i],
                                                                            FalseNumAll[th_i],
                                                                            FalseNumAll[
                                                                            th_i] / pixelsNumBatch))
    writelines.close()

    seg = list(Th_Seg).index(0.5)
    print('model: %s, epoch: %d, Th_Seg = %.4e, PD:[%d, %.5f], FA:[%d, %.4e], AUC:%.5f' % (
        args.model + args.loss_func, epoch + 1, Th_Seg[seg], TrueNumAll[seg],
        Pd_all[seg], FalseNumAll[seg], Fa_all[seg], auc_all))
    return
