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

    # def init_leaky(self):
    #     # Custom initialization, e.g., random normal values
    #     self.mem = torch.randn_like(self.mem, device=self.mem.device)
    #     return self.mem

    def forward(self, x, num_steps):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        feat_trace = []
        prob_trace = []
        spik_trace = []

        start_time = time.time()

        for _ in range(num_steps):
            # print(f"x.shape: {x.shape}")
            x = x.view(-1, self.num_channels * self.image_size * self.image_size)
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            feat_trace.append(spk1)
            prob_trace.append(mem2)
            spik_trace.append(spk2)

        elapsed_time = time.time() - start_time
        print(f"Forward pass time: {elapsed_time:.4f} seconds")

        return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)



class spikeLinearNet2(BasicModel):
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
        batch_size = x.shape[0]
        device = x.device

        # Initialize hidden states with correct batch size and device
        mem1 = torch.zeros(batch_size, 512, device=device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=device)

        # feat_trace = []
        # prob_trace = []
        # spik_trace = []

        # Pre-allocate output tensors
        spik_trace = torch.zeros(num_steps, batch_size, self.fc2.out_features, device=device)
        feat_trace = torch.zeros(num_steps, batch_size, 512, device=device)
        prob_trace = torch.zeros(num_steps, batch_size, self.fc2.out_features, device=device)

        start_time = time.time()
        
        for t in range(num_steps):
            x_flat = x.view(batch_size, -1)
            cur1 = self.fc1(x_flat)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # feat_trace.append(spk1)
            # prob_trace.append(mem2)
            # spik_trace.append(spk2)
            feat_trace[t] = spk1
            prob_trace[t] = mem2
            spik_trace[t] = spk2

        elapsed_time = time.time() - start_time
        print(f"Forward pass time: {elapsed_time:.4f} seconds")

        # return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)
        return spik_trace, feat_trace, prob_trace
