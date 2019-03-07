import torch
import torch.nn.functional  as F



class  fcn32(torch.nn.Module):

    def Conv_bn(self,input_channels,out_channels,kernels):

        block=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channels,out_channels=out_channels,kernel_size=kernels,padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        return block
    def Two_conv(self,input_channels,out_channels,kernels):
        block=torch.nn.Sequential(
            self.Conv_bn(input_channels,out_channels,kernels),
            self.Conv_bn(out_channels,out_channels,kernels)
        )
        return block
    def Three_conv(self,in_channels,out_channels,kernels):
        block=torch.nn.Sequential(
            self.Conv_bn(in_channels,out_channels,kernels),
            self.Conv_bn(in_channels,out_channels,kernels),
            self.Conv_bn(in_channels,out_channels,kernels)
        )
        return block


    def __init__(self,input_channels,classes):
        super(fcn32, self).__init__()
        self.conv1=self.Two_conv(input_channels=3,out_channels=input_channels,kernels=3)
        self.pool1=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=self.Two_conv(input_channels=input_channels,out_channels=input_channels*2,kernels=3)
        self.pool2=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3=self.Three_conv(in_channels=input_channels*2,out_channels=input_channels*4,kernels=3)
        self.pool3=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv4=self.Three_conv(in_channels=input_channels*4,out_channels=input_channels*8,kernels=3)
        self.pool4=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv5=self.Three_conv(in_channels=input_channels*8,out_channels=input_channels*16,kernels=3)
        self.pool5=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv6=self.Two_conv(input_channels=input_channels*16,out_channels=input_channels*32,kernels=3)
        self.up1=torch.nn.UpsamplingBilinear2d(scale_factor=32)
        self.out=torch.nn.Conv2d(kernel_size=1,in_channels=input_channels*32,out_channels=classes)
    def forward(self, x):
        en_conv1=self.conv1(x)
        en_pool1=self.pool1(en_conv1)
        en_conv2=self.conv2(en_pool1)
        en_pool2=self.pool2(en_conv2)
        en_conv3=self.conv3(en_pool2)
        en_pool3=self.pool3(en_conv3)
        en_conv4=self.conv4(en_pool3)
        en_pool4=self.pool4(en_conv4)
        en_conv5=self.conv5(en_pool4)
        en_pool5=self.pool5(en_conv5)
        en_conv6=self.conv6(en_pool5)
        de_up1=self.up1(en_conv6)
        de_out=self.out(de_up1)
        return de_out

if __name__=="__main__":

    model=fcn32(input_channels=32,classes=2)
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    optimizer.zero_grad()
    output=model(inputs)
    loss=criterion(output,label)
    loss.backward()
    optimizer.step()


