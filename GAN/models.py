import torch
from torch.nn.functional import relu, max_pool1d, sigmoid, log_softmax
import torch.nn as nn

device='cuda:1'
#----------------#
#   Generator    #
#----------------#

class UNetDownPath(nn.Module):
    def __init__(self, in_size, out_size, ksize=4, stride=2, normalize=True, dropout=0.0):
        super(UNetDownPath, self).__init__()
        layers = [nn.Conv1d(in_size, out_size, kernel_size=ksize,
                            stride=stride, bias=False, padding_mode='replicate')]
        if normalize:
            layers.append(nn.InstanceNorm1d(out_size))
        layers.append(nn.ReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUpPath(nn.Module):
    def __init__(self, in_size, out_size, ksize=4, stride=2, output_padding=0, dropout=0.0):
        super(UNetUpPath, self).__init__()
        layers = [
            nn.ConvTranspose1d(in_size, out_size, kernel_size=ksize,
                               stride=stride, output_padding=output_padding, bias=False),
            nn.InstanceNorm1d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

        self.pad = nn.ConstantPad1d((1,0,0,0),0)

    def forward(self, x, skip_input):
        x = self.model(x)

        if skip_input.shape[-1] == 310 or skip_input.shape[-1] == 623 or skip_input.shape[-1] == 1249:
            skip_input = self.pad(skip_input)

        x = torch.cat((x, skip_input), 1)

        return x

class Interpolate(nn.Module):
    def __init__(self, size):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        
    def forward(self, x):
        x = self.interp(x, size=self.size,mode='linear')
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDownPath(in_channels, 128, normalize=False)
        self.down2 = UNetDownPath(128, 256,dropout=0.5)
        self.down3 = UNetDownPath(256, 512, dropout=0.5)
        self.down4 = UNetDownPath(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUpPath(512, 512, output_padding=0, dropout=0.5)
        self.up2 = UNetUpPath(1024, 256, output_padding=1,dropout=0.5)
        self.up3 = UNetUpPath(512, 128, output_padding=1,dropout=0.5)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConstantPad1d((1, 1), 0),
            nn.Conv1d(256, out_channels, 4, padding=2,
                      padding_mode='replicate'),
            Interpolate(size=[500]),
            nn.Sigmoid(),
        )
        self.project_noise = nn.Linear(512, 512)
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        d4 = d4 + (0.1)*torch.randn(d4.shape).to(device)

        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.final(u3)

        return u4

class GeneratorUNetGenGan(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super(GeneratorUNetGenGan, self).__init__()

        self.down1 = UNetDownPath(in_channels, 128, normalize=False)
        self.down2 = UNetDownPath(128, 256,dropout=0.5)
        self.down3 = UNetDownPath(256, 512, dropout=0.5)
        self.down4 = UNetDownPath(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUpPath(517, 512, output_padding=0, dropout=0.5)
        self.up2 = UNetUpPath(1024, 256, output_padding=1,dropout=0.5)
        self.up3 = UNetUpPath(512, 128, output_padding=1,dropout=0.5)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConstantPad1d((1, 1), 0),
            nn.Conv1d(256, out_channels, 4, padding=2,
                      padding_mode='replicate'),
            Interpolate(size=[500]),
            nn.Sigmoid(),
        )
        self.project_noise = nn.Linear(512, 512)
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        rand_ = torch.rand(d4.shape[0],5,d4.shape[2]).to('cuda:0')
        d4 = torch.cat([d4,rand_], dim=1)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.final(u3)

        return u4

#-----------------#
#  Discriminator  #
#-----------------#

class Discriminator(nn.Module):
    def __init__(self, in_channels=12):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, ksize=6, stride=3, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv1d(in_filters, out_filters, ksize,
                                stride=stride, padding_mode='replicate')]
            if normalization:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 128, normalization=False),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),nn.Conv1d(512, 64, 4, bias=False, padding_mode='replicate'))

        self.flatten_1 = nn.Linear(in_features=896,out_features=448)
        self.relu_f1 = nn.ReLU(inplace=True)

        self.flatten_2 = nn.Linear(in_features=448,out_features=224)
        self.relu_f2 = nn.ReLU(inplace=True)

        self.flatten_3 = nn.Linear(in_features=224,out_features=1)
        self.relu_f3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.4)  #2.0
        
    def forward(self, data):
        x = self.model(data)
        x = torch.flatten(x,1)
        x = self.relu_f1(self.flatten_1(x))
        x = self.dropout(x)
        x = self.relu_f2(self.flatten_2(x))
        x = self.dropout(x)
        x = self.flatten_3(x)
        x = torch.squeeze(x)
        return x