''' 
Author: Aleksej Avramovic
Last update: 23/07/2024

The first step towards Out of Distribution detection

Spiking Resnet9 for MNIST
'''

import torch 
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import numpy as np 
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import sys
sys.path.append('../')
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchsummary import summary
import snntorch as snn
from snntorch import utils
import snntorch.functional as SF

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
    
    def accuracyS(self, model, numSteps, data, labels, device):
        with torch.no_grad():
            model.eval()
            data = data.to(device)
            labels = labels.to(device)
            spikes, _ = self.forward_pass(model, numSteps, data)
            acc = SF.accuracy_rate(spikes, labels) * spikes.size(1)
            total = spikes.size(1)

        return acc/total

    def fitS(self, model, nEpochs, opt, lossF, train_load, val_load, nSteps, device):
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
            # b_ = 0
            for batch, labels in train_load:
                # print("Training batch number ", b_)
                # b_ += 1
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
                a = self.accuracyS(model, nSteps, batch, labels, device)
                tAcc.append(a.item())

                del batch, labels

                # if b_ == 100:
                #     break
            # Training stats
            meanTA = sum(tAcc) / len(tAcc)
            meanTL = sum(tLoss) / len(tLoss)

            # Validation
            # Define lists to store validation loss and accuracy
            vLoss = []
            vAcc = list()
            model.eval()
            # v_ = 0
            for batch, labels in val_load:
                # print("Validation batch number ", v_)
                # v_ += 1
                batch = batch.to(device)
                labels = labels.to(device)
                spikes, _ = self.forward_pass(model, nSteps, batch)
                with torch.no_grad():
                    loss = lossF(spikes, labels)
                vLoss.append(loss.detach().item())
                a = self.accuracyS(model, nSteps, batch, labels, device)
                vAcc.append(a.item())  

                del batch, labels 
                # if v_ == 30:
                #     break
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
    
    def fitS_train(self, model, nEpochs, opt, lossF, train_load, nSteps, device):
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
            # b_ = 0
            for batch, labels in train_load:
                # print("Training batch number ", b_)
                # b_ += 1
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
                a = self.accuracyS(model, nSteps, batch, labels, device)
                tAcc.append(a.item())

                del batch, labels

                # if b_ == 100:
                #     break
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
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)                            # 64x28x28

        self.block2 = self.convBlock1(64, 128)                      # 128x28x28
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)                            # 128x28x28
        self.maxp2 = nn.MaxPool2d(2)                                # 128x14x14

        self.resBlock3_1 = self.convBlock1(128, 128)
        self.r3_lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.resBlock3_2 = self.convBlock1(128, 128)
        self.r3_lif2 = snn.Leaky(beta=beta, threshold=threshold)                         # 128x14x14
        
        self.block4 = self.convBlock1(128, 256)                     # 256x14x14
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold)
        self.maxp4 = nn.MaxPool2d(2)                                # 256x7x7

        self.block5 = self.convBlock1(256, 512)                     # 512x7x7
        self.lif5 = snn.Leaky(beta=beta, threshold=threshold)
        self.maxp5 = nn.MaxPool2d(2)                                # 512x3x3

        self.resBlock6_1 = self.convBlock1(512, 512)
        self.r6_lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.resBlock6_2 = self.convBlock1(512, 512)
        self.r6_lif2 = snn.Leaky(beta=beta, threshold=threshold)                         # 512x3x3

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

        return spk7, spk_out, mem7
    
    # spk7 512-D spikes
    # spk_out 10-D spikes
    # mem7 10-D memebrane voltage


# class FeatureExtractor(nn.Module):
#     '''
#     For custom SpikeResNet9Model only
#     '''
#     def __init__(self, network, beta, threshold):
#         super().__init__()
#         self.network = network
#         del self.network.lifOut
#         del self.network.fc7
#         self.network.lifOut = snn.Leaky(beta=beta, threshold=threshold, output=True)

#     def forward(self, x):
#         # Initialize hidden states and outputs at t=0
#         mem1 = self.network.lif1.init_leaky()
#         mem2 = self.network.lif2.init_leaky()
#         mem3_1 = self.network.r3_lif1.init_leaky()
#         mem3_2 = self.network.r3_lif2.init_leaky()
#         mem4 = self.network.lif4.init_leaky()
#         mem5 = self.network.lif5.init_leaky()
#         mem6_1 = self.network.r6_lif1.init_leaky()
#         mem6_2 = self.network.r6_lif2.init_leaky()
#         mem7 = self.network.lifOut.init_leaky()

#         cur1 = self.network.block1(x)
#         spk1, mem1 = self.network.lif1(cur1, mem1)
#         cur2 = self.network.block2(spk1)
#         spk2, mem2 = self.network.lif2(cur2, mem2)
#         spk2_2 = self.network.maxp2(spk2) # used for residual

#         cur3_1 = self.network.resBlock3_1(spk2_2)
#         spk3_1, mem3_1 = self.network.r3_lif1(cur3_1, mem3_1)
#         cur3_2 = self.network.resBlock3_2(spk3_1)
#         spk3_2, mem3_2 = self.network.r3_lif1(cur3_2, mem3_2)
#         spk_r1 = spk3_2 + spk2_2
        
#         cur4 = self.network.block4(spk_r1)
#         spk4, mem4 = self.network.lif4(cur4, mem4)
#         spk4_2 = self.network.maxp4(spk4) 

#         cur5 = self.network.block5(spk4_2)
#         spk5, mem5 = self.network.lif5(cur5, mem5)
#         spk5_2 = self.network.maxp5(spk5) # used for residual 

#         cur6_1 = self.network.resBlock6_1(spk5_2)
#         spk6_1, mem6_1 = self.network.r6_lif1(cur6_1, mem6_1)
#         cur6_2 = self.network.resBlock6_2(spk6_1)
#         spk6_2, mem6_2 = self.network.r6_lif1(cur6_2, mem6_2)
#         spk_r2 = spk6_2 + spk5_2

#         spk7 = self.network.amax7(spk_r2)
#         spk7 = self.network.flat(spk7)

#         out, mem7 = self.network.lifOut(spk7, mem7)

#         return out
    
# class CustomClassifier(nn.Module):
#     '''
#     For custom SpikeResNet9Model only
#     '''
#     def __init__(self, numberOfClasses, beta, threshold):
#         super().__init__()
#         self.fc = nn.Linear(512, numberOfClasses)
#         self.lif = snn.Leaky(beta=beta, threshold = threshold, reset_mechanism='none', output=True)

#     def forward(self, x):
#         mem = self.lif.init_leaky()
#         cur = self.fc(x)
#         out, mem = self.lif(cur, mem)
#         return mem

# # Combine feature extractor and classifier
# class CustomResNet(nn.Module):
#     def __init__(self, num_classes, network, beta, threshold):
#         super(CustomResNet, self).__init__()
#         self.feature_extractor = FeatureExtractor(network, beta, threshold)
#         self.classifier = CustomClassifier(num_classes, beta, threshold)
#     def forward(self, x):
#         x = self.feature_extractor(x)
#         y = self.classifier(x)
#         return (x, y)

