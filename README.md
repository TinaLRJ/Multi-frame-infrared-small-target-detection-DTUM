# Direction-coded Temporal U-shape Module for Multiframe Infrared Small Target Detection

Pytorch implementation of our Direction-coded Temporal U-shape Module (DTUM).&nbsp;[**[Paper]**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10321723)


## Requirements
- Python 3
- torch
- mmdet
- tqdm
- DCNv2
- scikit-image
<br><br>

## Datasets

NUDT-MIRSDT &nbsp; [[Baidu download dir]](https://pan.baidu.com/s/1pSN350eurMafLiHBQBnrPA?pwd=5whn) (Extraction code: 5whn)
is a synthesized dataset, which contains 120 sequences. We use 80 sequences for training and 20 sequences for test.
We divide the test set into two subsets according to their SNR ((0, 3], (3, 10)).

In the test set, targets in 8 sequences are so weak (SNR lower than 3). It is very challenging to detect these targets. The test set includes Sequence[47, 56, 59, 76, 92, 101, 105, 119].


## Train on NUDT-MIRSDT Dataset
```bash
python train.py --model 'ResUNet_DTUM' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True
python train.py --model 'DNANet_DTUM' --loss_func 'fullySup1' --train 1 --test 0 --fullySupervised True --SpatialDeepSup False
```
<br>


## Test on NUDT-MIRSDT Dataset
```bash
python train.py --model 'ResUNet_DTUM' --loss_func 'fullySup' --train 0 --test 1 --pth_path [trained model path]
```
<br>


## Train on IRDST Dataset (or other datasets)
```bash
python train.py --dataset IRDST --align --model 'ResUNet_DTUM' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True
```
If the background in a sequence is not aligned, the parameter of "align" needs to be set to True.
<br>


## Results and Trained Models

#### Quantative Results 

on NUDT-MIRSDT (SNR≤3)

| Model         | Pd (x10(-2))|  Fa (x10(-5)) | AUC ||
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| ALCNet  +DTUM | 56.144 | 0.931 | 0.9489|
| Res-UNet+DTUM | 91.682 | 2.369 | 0.9921 | [[Weights]](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM/blob/main/results/ResUNet_DTUM_SpatialDeepSupFalse_fullySup/ResUNet_DTUM.pth) |
| DNANet  +DTUM | 85.444 | 1.118 | 0.9882 |
| ISNet   +DTUM | 50.662 | 0.646 | 0.9482 |
| UIUNet  +DTUM | 72.023 | 1.916 | 0.9933 |


on NUDT-MIRSDT (SNR＞3)

| Model         | Pd (x10(-2))|  Fa (x10(-5)) | AUC ||
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| ALCNet  +DTUM | 99.500 | 2.370 | 0.9988|
| Res-UNet+DTUM | 100    | 3.415 | 0.9988 | [[Weights]](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM/blob/main/results/ResUNet_DTUM_SpatialDeepSupFalse_fullySup/ResUNet_DTUM.pth) |
| DNANet  +DTUM | 99.833 | 3.633 | 0.9988 |
| ISNet   +DTUM | 99.750 | 3.448 | 0.9988 |
| UIUNet  +DTUM | 99.833 | 3.578 | 0.9988 |


on NUDT-MIRSDT (all)

| Model         | Pd (x10(-2))|  Fa (x10(-5)) | AUC ||
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| ALCNet  +DTUM | 86.235 | 1.798 | 0.9818|
| Res-UNet+DTUM | 97.455 | 2.999 | 0.9964 | [[Weights]](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM/blob/main/results/ResUNet_DTUM_SpatialDeepSupFalse_fullySup/ResUNet_DTUM.pth) |
| DNANet  +DTUM | 95.431 | 2.620 | 0.9951 |
| ISNet   +DTUM | 84.731 | 2.334 | 0.9816 |
| UIUNet  +DTUM | 91.324 | 2.917 | 0.9972 |

## Citiation
```
@article{li2023direction,
  title={Direction-Coded Temporal U-Shape Module for Multiframe Infrared Small Target Detection.},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023}
}
```
<br>

## Contact
Welcome to raise issues or email to [liruojing@nudt.edu.cn](liruojing@nudt.edu.cn) for any question.
