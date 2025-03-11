import torch 
import torch.nn as nn
import sys
sys.path.append('../')
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import snntorch as snn
import snntorch.functional as SF
from models.spikeresnet import BasicModel


class spikeLinearNet1(BasicModel):
    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold, image_size=28):
        super().__init__(numberOfClasses)

        self.image_size = image_size
        self.num_channels = numberOfChannels

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.num_channels * self.image_size * self.image_size, 512, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.fc2 = nn.Linear(512, numberOfClasses, bias=False)
        # self.lif2 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")

    def forward(self, x, num_steps):

    # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        feat_trace = []
        prob_trace = []
        spik_trace = []

        for _ in range(num_steps):
            # print(f"x.shape: {x.shape}")
            x = x.view(-1, self.num_channels * self.image_size * self.image_size)
            # print(f"x.shape: {x.shape}")
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            feat_trace.append(spk1)
            prob_trace.append(mem2)
            spik_trace.append(spk2)

            return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)



# the old architecture
class spikeConvNN1(nn.Module):
    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold):
        # it was set to init threshold value 0.2
        super().__init__()

        self.conv1 = nn.Conv2d(numberOfChannels, 64, kernel_size=3, padding=1, stride=2)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.amax = nn.AdaptiveMaxPool2d(1)
        self.flat = nn.Flatten()    
        self.fc4 = nn.Linear(256, numberOfClasses)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero', output=True)

        # self.fc1 = nn.Linear(50*8*8, 500)
        # self.lif3 = snn.Leaky(beta=beta, threshold=threshold/2, reset_mechanism="zero")
        # self.fc2 = nn.Linear(500, 300)
        # self.lif4 = snn.Leaky(beta=beta, threshold=threshold/4, reset_mechanism="zero")
        # self.fc3 = nn.Linear(300, 10)
        # self.lif5 = snn.Leaky(beta=beta)

    def forward(self, x, num_steps):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        feat_trace = []
        prob_trace = []
        spik_trace = []

        for _ in range(num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk4 = self.amax(spk3)
            spk4 = self.flat(spk4)
            cur4 = self.fc4(spk4)
            spk_out, mem4 = self.lifOut(cur4, mem4)

            feat_trace.append(spk4)
            prob_trace.append(mem4)
            spik_trace.append(spk_out)

        return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)
