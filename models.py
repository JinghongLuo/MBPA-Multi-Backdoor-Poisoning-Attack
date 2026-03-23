import torch
import torch.nn as nn
from typing import Optional


def CalculateOutSize(blocks, channels, samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    x = torch.rand(1, 1, channels, samples)
    for block in blocks:
        block.eval()
        x = block(x)
    shape = x.shape[-2] * x.shape[-1]
    return shape


def LoadModel(model_name, n_classes, Chans, Samples, sap_frac=None):
    if model_name == 'EEGNet':
        model = EEGNet(n_classes=n_classes,
                       Chans=Chans,
                       Samples=Samples,
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25,
                       SAP_frac=sap_frac)
    elif model_name == 'DeepCNN':
        model = DeepConvNet(n_classes=n_classes,
                            Chans=Chans,
                            Samples=Samples,
                            dropoutRate=0.5,
                            SAP_frac=sap_frac)
    elif model_name == 'ShallowCNN':
        model=ShallowConvNet(n_classes=n_classes,
                            Chans=Chans,
                            Samples=Samples,
                            dropoutRate=0.5,
                            SAP_frac=sap_frac)
    else:
        raise 'No such model'
    return model


class SAP(nn.Module):
    def __init__(self, frac):
        super(SAP, self).__init__()
        self.frac = frac

    def forward(self, x):
        if self.frac is not None and   self.training ==False:
            shape = x.shape
            batch_size = shape[0]
            x_flat = x.reshape(batch_size, -1)
            n_features = x_flat.shape[1]

            K = max(1, int(self.frac * n_features))

            abs_x = torch.abs(x_flat)
            p = abs_x / (torch.sum(abs_x, dim=1, keepdim=True) + 1e-8)
            indices = torch.multinomial(p, K, replacement=True)

            mask = torch.zeros_like(x_flat)
            for i in range(batch_size):
                count = torch.bincount(indices[i], minlength=n_features)
                mask[i] = (count > 0).float()
            q = 1.0 - torch.pow(1.0 - p, K)  # 被选中至少一次的概率
            scaling = 1.0 / (q + 1e-8)

            result = x_flat * mask * scaling
            x=result.reshape(shape)
        return x


class EEGNet(nn.Module):
    """
    :param
    """

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: Optional[float] = 0.5,
                 SAP_frac: Optional[float] = None):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.SAP_frac = SAP_frac

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            SAP(frac=self.SAP_frac),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate)
        )

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            SAP(frac=self.SAP_frac),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate)
        )

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)

        return output
    #
    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if n == '3.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)

#
class DeepConvNet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5,
                 SAP_frac: Optional[float] = None):
        super(DeepConvNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.SAP_frac = SAP_frac
        # gutlaaa jhl
        # mlnb~
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5)),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(num_features=25), nn.ELU(),SAP(frac=self.SAP_frac),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=50), nn.ELU(),SAP(frac=self.SAP_frac),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate)
        )
        #wls
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=100), nn.ELU(),SAP(frac=self.SAP_frac),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate)
        )
        #
        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=100 *
                                  CalculateOutSize([self.block1, self.block2, self.block3],
                                                   self.Chans, self.Samples),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)

        return output

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.5)

#
class Activation(nn.Module):
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6))
        else:
            raise Exception('Invalid type !')

        return output


class ShallowConvNet(nn.Module):
    def __init__(
            self,
            n_classes: int,
            Chans: int,
            Samples: int,
            dropoutRate: Optional[float] = 0.5,
            SAP_frac:Optional[float]=None
    ):
        super(ShallowConvNet, self).__init__()
        #
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.SAP_frac = SAP_frac
        #
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 13)),
            nn.Conv2d(in_channels=40,
                      out_channels=40,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=40),

            Activation('square'),
            SAP(frac=self.SAP_frac),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            Activation('log'),
            SAP(frac=self.SAP_frac),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=40*CalculateOutSize([self.block1],
                                                   self.Chans, self.Samples),
                      out_features=self.n_classes,
                      bias=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.5)


class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.block = nn.Sequential(
            nn.Linear(in_features=self.input_dim,
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, feature):
        output = self.block(feature)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)


class FilterLayer(nn.Module):
    def __init__(self, order, band, fs):
        super(FilterLayer, self).__init__()

        B, A = butter(order, np.array(band) / (fs / 2), btype='bandpass')
        B = torch.from_numpy(B).type(torch.FloatTensor)
        A = torch.from_numpy(A).type(torch.FloatTensor)
        self.B = nn.Parameter(B, requires_grad=True)
        self.A = nn.Parameter(A, requires_grad=True)

    def forward(self, x):
        return F.lfilter(x, self.A, self.B)


