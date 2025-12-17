import torch 
import torch.nn as nn
import sys
sys.path.append('../')
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import snntorch as snn
import snntorch.functional as SF
from models.spikeresnet import BasicModel
import time

# Several old implementations
class _spikeLinearNet1(BasicModel):
    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold, image_size=28):
        super().__init__(numberOfClasses)

        self.image_size = image_size
        self.num_channels = numberOfChannels

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.num_channels * self.image_size * self.image_size, 512, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.fc2 = nn.Linear(512, numberOfClasses, bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")

    def init_mem(self, shape, device):
        return torch.randn(shape, device=device)

    def forward(self, x, num_steps):
        mem1 = self.init_mem((x.size(0), 512), x.device)
        mem2 = self.init_mem((x.size(0), self.fc2.out_features), x.device)

        feat_trace = []
        prob_trace = []
        spik_trace = []

        for _ in range(num_steps):
            x = x.view(-1, self.num_channels * self.image_size * self.image_size)
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            feat_trace.append(spk1)
            prob_trace.append(mem2)
            spik_trace.append(spk2)

        return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)


class __spikeLinearNet1(BasicModel):
    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold, image_size=28):
        super().__init__(numberOfClasses)

        self.image_size = image_size
        self.num_channels = numberOfChannels

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.num_channels * self.image_size * self.image_size, 512, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.fc2 = nn.Linear(512, numberOfClasses, bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        # Learnable initial membrane states (1D, will be broadcasted per batch)
        self.mem1_init = nn.Parameter(torch.randn(512) * 0.01)
        self.mem2_init = nn.Parameter(torch.randn(numberOfClasses) * 0.01)

    def init_mem(self, shape, device):
        return torch.randn(shape, device=device)

    def forward(self, x, num_steps):
        # mem1 = self.init_mem((x.size(0), 512), x.device)
        # mem2 = self.init_mem((x.size(0), self.fc2.out_features), x.device)

        mem1 = self.mem1_init.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        mem2 = self.mem2_init.unsqueeze(0).expand(x.shape[0], -1).to(x.device)

        feat_trace = []
        prob_trace = []
        spik_trace = []

        for _ in range(num_steps):
            x = x.view(-1, self.num_channels * self.image_size * self.image_size)
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            feat_trace.append(spk1)
            prob_trace.append(mem2)
            spik_trace.append(spk2)

        return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)


class spikeLinearNet1(BasicModel):
    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold, feature_size=28):
        super().__init__(numberOfClasses)
        self.features = int(((feature_size -2)/2)-2)
        self.averaging = int((feature_size - 2)/2)

        self.conv1 = nn.Conv2d(numberOfChannels, 20, kernel_size=3, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.avg1 = nn.AdaptiveAvgPool2d(self.averaging)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3, bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.fc3 = nn.Linear(self.features * self.features * 50, 500, bias=False)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold/2, reset_mechanism="zero") 
        self.fc4 = nn.Linear(500, 300, bias=False)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold/4, reset_mechanism='zero')
        self.fc5 = nn.Linear(300, numberOfClasses, bias=False)
        self.lif5 = snn.Leaky(beta=beta) # this one should be only leaky integrate, not fire
        self.mem1_init = nn.Parameter(torch.randn(20) * 0.01)
        self.mem2_init = nn.Parameter(torch.randn(50) * 0.01)
        self.mem3_init = nn.Parameter(torch.randn(500) * 0.01)
        self.mem4_init = nn.Parameter(torch.randn(300) * 0.01)
        self.mem5_init = nn.Parameter(torch.randn(numberOfClasses) * 0.01)

    def forward(self, x, num_steps):

        mem1 = self.mem1_init.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        mem2 = self.mem2_init.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        mem3 = self.mem3_init.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        mem4 = self.mem4_init.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        mem5 = self.mem1_init.unsqueeze(0).expand(x.shape[0], -1).to(x.device)

        feat_trace = []
        prob_trace = []
        spik_trace = []

        for _ in range(num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            # print(f"Conv1 spk1.shape: {spk1.shape}")
            spk1 = self.avg1(spk1)
            # print(f"Avg1 spk1.shape: {spk1.shape}")

            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            # print(f"Conv2 spk2.shape: {spk2.shape}")
            spk2 = spk2.view(-1, self.features * self.features * 50)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc5(spk4)
            spk_out, mem5 = self.lif5(cur5, mem5)

            feat_trace.append(spk4)
            prob_trace.append(mem5)
            spik_trace.append(spk_out)

        return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)



# What do we do next