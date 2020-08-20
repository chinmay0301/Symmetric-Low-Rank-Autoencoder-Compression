import torch.nn as nn
import torch.nn.functional as F

from .prune import PruningModule, MaskedLinear

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

class AE(PruningModule):
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        linear = MaskedLinear if kwargs['mask'] else nn.Linear
        self.encoder_l1 = linear(
            in_features=kwargs["input_shape"], out_features=256
        )
        self.encoder_l2 = linear(
            in_features=256, out_features=128
        )
        self.encoder_l3 = linear(
            in_features=128, out_features=64
        )
        
        self.decoder_l3 = linear(
            in_features=64, out_features=128
        )
        self.decoder_l2 = linear(
            in_features=128, out_features=256
        )
        self.decoder_l1 = linear(
            in_features=256, out_features=kwargs["input_shape"]
        )

    def forward(self, x):
        x = x.view(-1, 784)
        activation = self.encoder_l1(x)
        activation = F.relu(activation)
        activation = self.encoder_l2(activation)
        activation = F.relu(activation)

        code = self.encoder_l3(activation)
        code = F.relu(code)
        
        activation = self.decoder_l3(code)
        activation = F.relu(activation)
        activation = self.decoder_l2(activation)
        activation = F.relu(activation)
        activation = self.decoder_l1(activation)
        reconstructed = F.relu(activation)
        return reconstructed

class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else Linear
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.fc1 = linear(120, 84)
        self.fc2 = linear(84, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3
        x = self.conv3(x)
        x = F.relu(x)

        # Fully-connected
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
