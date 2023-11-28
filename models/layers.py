import torch
from torch import nn


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace = True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = out + residual
        return out



class DTUM(nn.Module):    # final version
    def __init__(self, in_channels, num_classes, num_frames):
        super(DTUM, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), return_indices=True)
        # self.pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), return_indices=True, ceil_mode=False)
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='nearest')
        self.relu = nn.ReLU(inplace=True)

        inch = in_channels
        pad = int((num_frames-1)/2)
        self.bn0 = nn.BatchNorm3d(inch)
        self.conv1_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn1_1 = nn.BatchNorm3d(inch)
        self.conv2_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn2_1 = nn.BatchNorm3d(inch)
        self.conv3_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn3_1 = nn.BatchNorm3d(inch)
        self.conv4_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn4_1 = nn.BatchNorm3d(inch)

        self.conv3_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn3_2 = nn.BatchNorm3d(inch)
        self.conv2_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn2_2 = nn.BatchNorm3d(inch)
        self.conv1_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(0,0,0))
        self.bn1_2 = nn.BatchNorm3d(inch)

        self.final = nn.Sequential(
            nn.Conv3d(in_channels=2*inch, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.Dropout3d(0.5),
            nn.Conv3d(in_channels=32, out_channels=num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        )

    def direction(self, arr):
        b,c,t,m,n = arr.size()
        arr[:, :, 1:, :, :] = arr[:, :, 1:, :, :] - m * 2 * n * 2
        arr[:, :, 2:, :, :] = arr[:, :, 2:, :, :] - m * 2 * n * 2
        arr[:, :, 3:, :, :] = arr[:, :, 3:, :, :] - m * 2 * n * 2
        arr[:, :, 4:, :, :] = arr[:, :, 4:, :, :] - m * 2 * n * 2

        arr_r_l = arr % 2  # right 1; left 0     [0 1; 0 1]
        up_down = torch.Tensor(range(0,m)).cuda(arr.device) * n*2*2  #.transpose(0,1)
        up_down = up_down.repeat_interleave(n).reshape(m,n)
        arr1 = arr.float() - up_down.reshape([1,1,1,m,n])
        arr_u_d = (arr1 >= n*2).float() * 2  # up 0; down 1  [0 0; 2 2]
        arr_out = arr_r_l.float() + arr_u_d   # [0 1; 2 3]
        arr_out = (arr_out - 1.5)       # [-1.5 -0.5; 0.5 1.5]

        return arr_out


    def forward(self, x):

        x = self.relu(self.bn0(x))

        x_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        xp_1, ind = self.pool(x_1)
        x_2 = self.relu(self.bn2_1(torch.abs(self.conv2_1(xp_1 * self.direction(ind)))))
        xp_2, ind = self.pool(x_2)
        x_3 = self.relu(self.bn3_1(torch.abs(self.conv3_1(xp_2 * self.direction(ind)))))
        xp_3, ind = self.pool(x_3)
        x_4 = self.relu(self.bn4_1(torch.abs(self.conv4_1(xp_3 * self.direction(ind)))))

        o_3 = self.relu(self.bn3_2(self.conv3_2(torch.cat([self.up(x_4),x_3], dim=1))))
        o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(o_3),x_2], dim=1)))).detach()
        o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1))))

        x_out = self.final(torch.cat([o_1, torch.unsqueeze(x[:,:,-1,:,:],2)], dim=1))

        return x_out


