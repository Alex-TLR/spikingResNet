''' 
Author: Aleksej Avramovic
Last update: 27/08/2024

The first step towards spiking In Distribution / Out of Distribution detection

SpikeResNet9Model: Spiking-Resnet9 for MNIST, from the scratch
'''

# TODO: make identity mapping
# TODO: make generic Resenet model

import torch 
import numpy as np 
import torch.nn as nn
import sys
sys.path.append('../')
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import snntorch as snn
from snntorch import utils
import snntorch.functional as SF
from snntorch import spikegen


# Utility functions
###################

def showBatch(inputData):
    for images, labels in inputData:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break

# Define basic model
####################

class BasicModel(nn.Module):

    def __init__(self, nClasses):
        super().__init__()
        self.numberOfClasses = nClasses
    
    def progressBar(self, iter, total, prefix = '', suffix = '', length = 30, fill = '#'):
        percent = f'{100 * (iter / (float(total))):.1f}'
        filled = int(length * iter // total)
        bar = fill * filled + '_' * (length - filled) + ' ' + percent
        sys.stdout.write('\r%s |%s%% %s' % (prefix, bar, suffix))
        sys.stdout.flush()
        return None
    
    def forward_pass(self, model, numSteps, data):
        mem_trace = []
        spk_trace = []
        utils.reset(model)

        for _ in range(numSteps):
            spk_out, _, mem_out = model(data)
            spk_trace.append(spk_out)
            mem_trace.append(mem_out)

        spk_trace = torch.stack(spk_trace)
        mem_trace = torch.stack(mem_trace)

        return spk_trace, mem_trace
    
    def forward_pass_rate(self, model, numSteps, data):
        mem_trace = []
        spk_trace = []
        utils.reset(model)

        spike_data = spikegen.rate(data, num_steps=numSteps)
        for i in range(numSteps):
            spk_out, _, mem_out = model(spike_data[i])
            spk_trace.append(spk_out)
            mem_trace.append(mem_out)

        spk_trace = torch.stack(spk_trace)
        mem_trace = torch.stack(mem_trace)

        return spk_trace, mem_trace
    
    def accuracy_spike(self, model, numSteps, data, labels, device):

        with torch.no_grad():
            model.eval()
            data = data.to(device)
            labels = labels.to(device)
            spikes, _ = self.forward_pass(model, numSteps, data)
            acc = SF.accuracy_rate(spikes, labels) * spikes.size(1)
            total = spikes.size(1)

        return acc/total

    def fit_spike(self, model, nEpochs, opt, lossF, train_load, val_load, nSteps, device):
        '''
        nEpochs:      number of epochs for training
        opt:          optimizer
        lossf:        loss function
        train_load:   loader for train data
        val_load:     loader for validation data
        nSteps:       number of steps in one exitation !
        history:      list of statistics for all epochs (appended in each epoch)
        '''
        history = []

        for i in range(nEpochs):
            # model.train()
            # Define lists to store training loss and accuracy
            tLoss = []
            tAcc = list()
            for batch, labels in train_load:
                batch = batch.to(device)
                labels = labels.to(device)
                model.train()
                # Generate predictions/ forward pass
                spikes, _ = self.forward_pass(model, nSteps, batch)
                # Calculate loss
                loss = lossF(spikes, labels) 
                tLoss.append(loss.detach().item())
                # Set opt grad
                opt.zero_grad()
                # Update weights
                loss.backward()
                # Update opt
                opt.step()
                # Check train accuracy
                a = self.accuracy_spike(model, nSteps, batch, labels, device)
                tAcc.append(a.item())

                del batch, labels

            # Training stats
            meanTA = sum(tAcc) / len(tAcc)
            meanTL = sum(tLoss) / len(tLoss)

            # Validation
            # Define lists to store validation loss and accuracy
            vLoss = []
            vAcc = list()
            model.eval()
            for batch, labels in val_load:
                batch = batch.to(device)
                labels = labels.to(device)
                spikes, _ = self.forward_pass(model, nSteps, batch)
                with torch.no_grad():
                    loss = lossF(spikes, labels)
                vLoss.append(loss.detach().item())
                a = self.accuracy_spike(model, nSteps, batch, labels, device)
                vAcc.append(a.item())  

                del batch, labels 

            # Validation stats
            meanVA = sum(vAcc) / len(vAcc)
            meanVL = sum(vLoss) / len(vLoss)            

            # Make progress bar
            suffixArray = ' ' + 'Training loss: ' + f'{meanTL:.2f} ' + 'Training accuracy: ' + f'{meanTA:.2f} ' + \
            'Validation loss: ' + f'{meanVL:.2f} ' + 'Validation accuracy: ' + f'{meanVA:.2f}'

            self.progressBar(i + 1, nEpochs, prefix = 'Progress: ', suffix = suffixArray, length = 40, fill = '#')
            currentHistory = [meanTL, meanTA, meanVL, meanVA]
            history.append(currentHistory)

        print('\n')
        return history
    
    def fit_spike_full_train(self, model, nEpochs, opt, lossF, train_load, nSteps, device):
        '''
        nEpochs:      number of epochs for training
        opt:          optimizer
        lossf:        loss function
        train_load:   loader for train data
        nSteps:       number of steps in one exitation !
        history:      list of statistics for all epochs (appended in each epoch)
        '''
        history = []

        for i in range(nEpochs):
            # model.train()
            # Define lists to store training loss and accuracy
            tLoss = []
            tAcc = list()
            for batch, labels in train_load:
                batch = batch.to(device)
                labels = labels.to(device)
                model.train()
                # Generate predictions/ forward pass
                spikes, _ = self.forward_pass(model, nSteps, batch)
                # Calculate loss
                loss = lossF(spikes, labels) 
                tLoss.append(loss.detach().item())
                # Set opt grad
                opt.zero_grad()
                # Update weights
                loss.backward()
                # Update opt
                opt.step()
                # Check train accuracy
                a = self.accuracy_spike(model, nSteps, batch, labels, device)
                tAcc.append(a.item())

                del batch, labels

            # Training stats
            meanTA = sum(tAcc) / len(tAcc)
            meanTL = sum(tLoss) / len(tLoss)           

            # Make progress bar
            suffixArray = ' ' + 'Training loss: ' + f'{meanTL:.2f} ' + 'Training accuracy: ' + f'{meanTA:.2f} '

            self.progressBar(i + 1, nEpochs, prefix = 'Progress: ', suffix = suffixArray, length = 40, fill = '#')
            currentHistory = [meanTL, meanTA]
            history.append(currentHistory)

        print('\n')
        return history
    

# Define Resnet model
#####################

class SpikeResNet9Model(BasicModel):

    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold):
        super().__init__(numberOfClasses)

        self.block1 = self.convBlock1(numberOfChannels, 64)         # 64x28x28
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)       # 64x28x28

        self.block2 = self.convBlock1(64, 128)                      # 128x28x28
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)       # 128x28x28
        self.maxp2 = nn.MaxPool2d(2)                                # 128x14x14

        self.resBlock3_1 = self.convBlock1(128, 128)
        self.r3_lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.resBlock3_2 = self.convBlock1(128, 128)
        self.r3_lif2 = snn.Leaky(beta=beta, threshold=threshold)    # 128x14x14
        
        self.block4 = self.convBlock1(128, 256)                     # 256x14x14
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold)
        self.maxp4 = nn.MaxPool2d(2)                                # 256x7x7

        self.block5 = self.convBlock1(256, 512)                     # 512x7x7
        self.lif5 = snn.Leaky(beta=beta, threshold=threshold)
        self.maxp5 = nn.MaxPool2d(2)                                # 512x3x3

        self.resBlock6_1 = self.convBlock1(512, 512)
        self.r6_lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.resBlock6_2 = self.convBlock1(512, 512)
        self.r6_lif2 = snn.Leaky(beta=beta, threshold=threshold)    # 512x3x3

        self.amax7 = nn.AdaptiveMaxPool2d(1)
        self.flat = nn.Flatten()
        self.fc7 = nn.Linear(512, numberOfClasses)
        self.lifOut = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='none', output=True)

    def convBlock1(self, input, output):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, padding=1, bias=False),
                  nn.BatchNorm2d(num_features=output)]
        return nn.Sequential(*layers)
    
    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3_1 = self.r3_lif1.init_leaky()
        mem3_2 = self.r3_lif2.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6_1 = self.r6_lif1.init_leaky()
        mem6_2 = self.r6_lif2.init_leaky()
        mem7 = self.lifOut.init_leaky()

        cur1 = self.block1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.block2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        spk2_2 = self.maxp2(spk2) # used for residual

        cur3_1 = self.resBlock3_1(spk2_2)
        spk3_1, mem3_1 = self.r3_lif1(cur3_1, mem3_1)
        cur3_2 = self.resBlock3_2(spk3_1)
        spk3_2, mem3_2 = self.r3_lif1(cur3_2, mem3_2)
        spk_r1 = spk3_2 + spk2_2
        
        cur4 = self.block4(spk_r1)
        spk4, mem4 = self.lif4(cur4, mem4)
        spk4_2 = self.maxp4(spk4) 

        cur5 = self.block5(spk4_2)
        spk5, mem5 = self.lif5(cur5, mem5)
        spk5_2 = self.maxp5(spk5) # used for residual 

        cur6_1 = self.resBlock6_1(spk5_2)
        spk6_1, mem6_1 = self.r6_lif1(cur6_1, mem6_1)
        cur6_2 = self.resBlock6_2(spk6_1)
        spk6_2, mem6_2 = self.r6_lif1(cur6_2, mem6_2)
        spk_r2 = spk6_2 + spk5_2

        spk7 = self.amax7(spk_r2)
        spk7 = self.flat(spk7)
        cur7 = self.fc7(spk7)
        spk_out, mem7 = self.lifOut(cur7, mem7)

        return spk_out, spk7, mem7
    
    # spk7 512-D spikes
    # spk_out 10-D spikes
    # mem7 10-D memebrane voltage
