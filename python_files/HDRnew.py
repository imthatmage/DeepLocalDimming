import torch
from torchvision import models


class UpSample(torch.nn.Module):
    def __init__(self, in_channels, s_out_channels=None, mid_layer_channels=None, size=None, div_factor=2):
        super(UpSample, self).__init__()

        self.size = size

        if s_out_channels is None:
            s_out_channels = in_channels//(2*div_factor)
        if mid_layer_channels is None:
            mid_layer_channels = in_channels//div_factor

        self.seq0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels//div_factor,
                            kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(in_channels//div_factor),
            torch.nn.LeakyReLU(),  # May be another activate function
            torch.nn.Conv2d(in_channels//div_factor, mid_layer_channels,
                            kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(mid_layer_channels)
        )

        self.conv = torch.nn.Conv2d(
            mid_layer_channels, s_out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
        x = self.seq0(x)
        if self.size is None:
            x = torch.nn.functional.interpolate(x, scale_factor=2,
                                                mode='bilinear', align_corners=False)
        else:
            x = torch.nn.functional.interpolate(
                x, size=self.size, mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = torch.cat((x[:, :], y[:, :]), dim=1)
        return x


# resnet18 for encoderI
resnet18 = models.resnet18(pretrained=True)
# for param in resnet18.parameters():
#    param.requires_grad = False


class HDRnet(torch.nn.Module):
    def __init__(self, n_hidden_neurons=64):
        super(HDRnet, self).__init__()
        self.act_func = torch.nn.LeakyReLU()
        # first freezed layer
        self.counter = 59
        resnet_lst = list(resnet18.children())

        # ENCODER. Here we are using resnet18 as a base for encoder
        self.encode0 = torch.nn.Sequential(
            *resnet_lst[0:3]
        )
        self.encode1 = torch.nn.Sequential(
            *resnet_lst[3:5]
        )
        self.encode2 = torch.nn.Sequential(
            *resnet_lst[5]
        )
        self.encode3 = torch.nn.Sequential(
            *resnet_lst[6]
        )
        self.encode4 = torch.nn.Sequential(
            *resnet_lst[7]
        )
        # END ENCODER

        # DECODER
        # 1
        self.decode0 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=False),
            torch.nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU()
        )
        # CONCATENATE HERE
        # 2
        self.decode1 = UpSample(512, size=(135, 240))
        self.decode2 = UpSample(256)
        self.decode3 = UpSample(128, div_factor=1)
        self.decode4 = UpSample(128, s_out_channels=4,
                                mid_layer_channels=32, div_factor=2)
        self.decode5 = torch.nn.Sequential(
            # 6 should be 7
            torch.nn.Conv2d(7, 4, kernel_size=3, stride=1,
                            padding=1, bias=False),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(4, 1, kernel_size=3, stride=1,
                            padding=1, bias=False),
            torch.nn.Sigmoid()
        )
        # 3

    def forward(self, sup0):
        x = self.encode0(sup0)
        sup1 = x
        x = self.encode1(x)
        sup2 = x
        x = self.encode2(x)
        sup3 = x
        x = self.encode3(x)
        sup4 = x
        x = self.encode4(sup4)

        x = self.decode0(x)
        x = torch.cat((x, sup4), dim=1)

        x = self.decode1(x, sup3)
        x = self.decode2(x, sup2)
        x = self.decode3(x, sup1)
        x = self.decode4(x, sup0)

        x = self.decode5(x)

        return x
