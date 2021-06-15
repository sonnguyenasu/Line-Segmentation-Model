import torch.nn as nn
import torch
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.softmax = nn.Softmax2d()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):

        # encoder
        x = self.conv1(x)
        out1 = self.leaky_reLU(x)
        x = out1
        size1 = x.size()
        x, indices1 = self.pool(x)

        x = self.conv2(x)
        out2 = self.leaky_reLU(x)
        x = out2
        size2 = x.size()
        x, indices2 = self.pool(x)

        x = self.conv3(x)
        out3 = self.leaky_reLU(x)
        x = out3
        size3 = x.size()
        x, indices3 = self.pool(x)

        x = self.conv4(x)
        out4 = self.leaky_reLU(x)
        x = out4
        size4 = x.size()
        x, indices4 = self.pool(x)

        ######################
        x = self.conv5(x)
        x = self.leaky_reLU(x) 

        x = self.conv6(x)
        x = self.leaky_reLU(x)
        ######################

        # decoder
        x = self.unpool(x, indices4, output_size=size4)
        x = self.conv7(torch.cat((x, out4), 1))
        x = self.leaky_reLU(x)

        x = self.unpool(x, indices3, output_size=size3)
        x = self.conv8(torch.cat((x, out3), 1))
        x = self.leaky_reLU(x)

        x = self.unpool(x, indices2, output_size=size2)
        x = self.conv9(torch.cat((x, out2), 1))
        x = self.leaky_reLU(x)

        x = self.unpool(x, indices1, output_size=size1)
        x = self.conv10(torch.cat((x, out1), 1))
        x = self.softmax(x)

        return x

if __name__ == '__main__':
    x = torch.rand(2,3,640,640)
    model = AE()
    y = model(x)
    print(y.shape)
