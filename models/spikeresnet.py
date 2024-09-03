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
                                                                                # Input size
    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold):     # 28x28
        super().__init__(numberOfClasses)

        self.block1 = self.convBlock1(numberOfChannels, 64)                     # 64x28x28
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)                   # 64x28x28

        self.block2 = self.convBlock1(64, 128)                                  # 128x28x28
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)                   # 128x28x28
        self.maxp2 = nn.MaxPool2d(2)                                            # 128x14x14

        self.resBlock3_1 = self.convBlock1(128, 128)
        self.r3_lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.resBlock3_2 = self.convBlock1(128, 128)
        self.r3_lif2 = snn.Leaky(beta=beta, threshold=threshold)                # 128x14x14
        
        self.block4 = self.convBlock1(128, 256)                                 # 256x14x14
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold)
        self.maxp4 = nn.MaxPool2d(2)                                            # 256x7x7

        self.block5 = self.convBlock1(256, 512)                                 # 512x7x7
        self.lif5 = snn.Leaky(beta=beta, threshold=threshold)
        self.maxp5 = nn.MaxPool2d(2)                                            # 512x3x3

        self.resBlock6_1 = self.convBlock1(512, 512)
        self.r6_lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.resBlock6_2 = self.convBlock1(512, 512)
        self.r6_lif2 = snn.Leaky(beta=beta, threshold=threshold)                # 512x3x3

        self.amax7 = nn.AdaptiveMaxPool2d(1)                                    # 512x1x1
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

# 18 layers
class SpikeResNet18Model(BasicModel):                                           # Input size

    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold):     # 32x32
        super().__init__(numberOfClasses)

        self.block1 = self.convBlock(numberOfChannels, 64)                      # 64x32x32
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)                   # 64x32x32

        # The residual super-block
        # Contains two blocks with skip connections ( 4 conv blocks)
        self.resBlock2_1 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_lif1 = snn.Leaky(beta=beta, threshold=threshold) 
        self.resBlock2_2 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.resBlock2_3 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_lif3 = snn.Leaky(beta=beta, threshold=threshold) 
        self.resBlock2_4 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_lif4 = snn.Leaky(beta=beta, threshold=threshold)  
        
        # Downsample block
        self.downsample3 = self.convBlock(64, 128, kernel_size=1, stride=2, padding=0)

        # The residual super-block
        # Contains two blocks with skip connections ( 4 conv blocks)
        self.resBlock4_1 = self.convBlock(64, 128, kernel_size=3, stride=2)     # 64x32x32
        self.r4_lif1 = snn.Leaky(beta=beta, threshold=threshold) 
        self.resBlock4_2 = self.convBlock(128, 128)                             # 128x16x16
        # Here is the place for skip connection
        self.r4_lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.resBlock4_3 = self.convBlock(128, 128)                             # 128x16x16
        self.r4_lif3 = snn.Leaky(beta=beta, threshold=threshold) 
        self.resBlock4_4 = self.convBlock(128, 128)                             # 128x16x16
        # Here is the place for skip connection
        self.r4_lif4 = snn.Leaky(beta=beta, threshold=threshold) 

        # Downsample block
        self.downsample5 = self.convBlock(128, 256, kernel_size=1, stride=2, padding=0)

        # The residual super-block
        # Contains two blocks with skip connections ( 4 conv blocks)
        self.resBlock6_1 = self.convBlock(128, 256, kernel_size=3, stride=2)    # 128x16x16
        self.r6_lif1 = snn.Leaky(beta=beta, threshold=threshold) 
        self.resBlock6_2 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.resBlock6_3 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_lif3 = snn.Leaky(beta=beta, threshold=threshold) 
        self.resBlock6_4 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_lif4 = snn.Leaky(beta=beta, threshold=threshold) 

        # Downsample block
        self.downsample7 = self.convBlock(256, 512, kernel_size=1, stride=2, padding=0)

        # The residual super-block
        # Contains two blocks with skip connections ( 4 conv blocks)
        self.resBlock8_1 = self.convBlock(256, 512, kernel_size=3, stride=2)    # 256x8x8
        self.r8_lif1 = snn.Leaky(beta=beta, threshold=threshold) 
        self.resBlock8_2 = self.convBlock(512, 512)                             # 512x4x4
        # Here is the place for skip connection
        self.r8_lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.resBlock8_3 = self.convBlock(512, 512)                             # 512x4x4
        self.r8_lif3 = snn.Leaky(beta=beta, threshold=threshold) 
        self.resBlock8_4 = self.convBlock(512, 512)                             # 512x4x4
        # Here is the place for skip connection
        self.r8_lif4 = snn.Leaky(beta=beta, threshold=threshold) 

        self.amax9 = nn.AdaptiveMaxPool2d(1)
        self.flat = nn.Flatten()
        self.fc10 = nn.Linear(512, numberOfClasses)
        self.lifOut = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='none', output=True)

    def convBlock(self, input, output, kernel_size=3, stride=1, padding=1):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(num_features=output)]
        return nn.Sequential(*layers)
    
    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2_1 = self.r2_lif1.init_leaky()
        mem2_2 = self.r2_lif2.init_leaky()
        mem2_3 = self.r2_lif3.init_leaky()
        mem2_4 = self.r2_lif4.init_leaky()
        mem4_1 = self.r4_lif1.init_leaky()
        mem4_2 = self.r4_lif2.init_leaky()
        mem4_3 = self.r4_lif3.init_leaky()
        mem4_4 = self.r4_lif4.init_leaky()
        mem6_1 = self.r6_lif1.init_leaky()
        mem6_2 = self.r6_lif2.init_leaky()
        mem6_3 = self.r6_lif3.init_leaky()
        mem6_4 = self.r6_lif4.init_leaky()
        mem8_1 = self.r8_lif1.init_leaky()
        mem8_2 = self.r8_lif2.init_leaky()
        mem8_3 = self.r8_lif3.init_leaky()
        mem8_4 = self.r8_lif4.init_leaky()
        mem_out = self.lifOut.init_leaky()

        cur1 = self.block1(x)                       # Conv block gives current for the following 
        spk1, mem1 = self.lif1(cur1, mem1)          # Leaky gives spk1 and membrane voltage
        # spk1 is used for skip connection

        # Block input conv (64, 64)
        cur2_1 = self.resBlock2_1(spk1)
        spk2_1, mem2_1 = self.r2_lif1(cur2_1, mem2_1)
        cur2_2 = self.resBlock2_2(spk2_1)
        spk2_2, mem2_2 = self.r2_lif2(cur2_2, mem2_2)
        # make skip connection, ADD :)
        spk_r2_1 = spk2_2 + spk1
        # spk_r2_1 is used for skip connection
        cur2_3 = self.resBlock2_3(spk_r2_1)
        spk2_3, mem2_3 = self.r2_lif3(cur2_3, mem2_3)
        cur2_4 = self.resBlock2_4(spk2_3)
        spk2_4, mem2_4 = self.r2_lif4(cur2_4, mem2_4)
        # make skip connection, ADD :)
        spk_r2_2 = spk2_4 + spk_r2_1
        # print(f'spk_r2_2.shape is {spk_r2_2.shape}')

        # Block input conv (64, 128)
        identity_4 = self.downsample3(spk_r2_2)
        # print(f'identity_4.shape is {identity_4.shape}')
        cur4_1 = self.resBlock4_1(spk_r2_2)
        spk4_1, mem4_1 = self.r4_lif1(cur4_1, mem4_1)
        cur4_2 = self.resBlock4_2(spk4_1)
        spk4_2, mem4_2 = self.r4_lif2(cur4_2, mem4_2)
        # make skip connection
        spk_r4_1 = spk4_2 + identity_4
        cur4_3 = self.resBlock4_3(spk_r4_1)
        spk4_3, mem4_3 = self.r4_lif3(cur4_3, mem4_3)
        cur4_4 = self.resBlock4_4(spk4_3)
        spk4_4, mem4_4 = self.r4_lif4(cur4_4, mem4_4)
        # make skip connection
        spk_r4_2 = spk4_4 + spk_r4_1

        # Block input conv (128, 256)
        identity_6 = self.downsample5(spk_r4_2)
        cur6_1 = self.resBlock6_1(spk_r4_2)
        spk6_1, mem6_1 = self.r6_lif1(cur6_1, mem6_1)
        cur6_2 = self.resBlock6_2(spk6_1)
        spk6_2, mem6_2 = self.r6_lif2(cur6_2, mem6_2)
        # make skip connection
        spk_r6_1 = spk6_2 + identity_6
        cur6_3 = self.resBlock6_3(spk_r6_1)
        spk6_3, mem6_3 = self.r6_lif3(cur6_3, mem6_3)
        cur6_4 = self.resBlock6_4(spk6_3)
        spk6_4, mem6_4 = self.r6_lif4(cur6_4, mem6_4)
        # make skip connection
        spk_r6_2 = spk6_4 + spk_r6_1

        # Block input conv (256, 512)
        identity_8 = self.downsample7(spk_r6_2)
        cur8_1 = self.resBlock8_1(spk_r6_2)
        spk8_1, mem8_1 = self.r8_lif1(cur8_1, mem8_1)
        cur8_2 = self.resBlock8_2(spk8_1)
        spk8_2, mem8_2 = self.r8_lif2(cur8_2, mem8_2)
        # make skip connection
        spk_r8_1 = spk8_2 + identity_8
        cur8_3 = self.resBlock8_3(spk_r8_1)
        spk8_3, mem8_3 = self.r8_lif3(cur8_3, mem8_3)
        cur8_4 = self.resBlock8_4(spk8_3)
        spk8_4, mem8_4 = self.r8_lif4(cur8_4, mem8_4)
        # make skip connection
        spk_r8_2 = spk8_4 + spk_r8_1

        spk9 = self.amax9(spk_r8_2)
        spk9 = self.flat(spk9)
        cur9 = self.fc10(spk9)
        spk_out, mem_out = self.lifOut(cur9, mem_out)

        return spk_out, spk9, mem_out
