''' 
Author: Aleksej Avramovic
Last update: 27/08/2024

The first step towards spiking In Distribution / Out of Distribution detection

SpikeResNet9Model: Spiking-Resnet9 for MNIST, from the scratch
'''

# TODO: make generic Resenet model

import torch 
import torch.nn as nn
import sys
sys.path.append('../')
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import snntorch as snn
import snntorch.functional as SF


# Utility functio06ns
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
    
    def accuracy_spike(self, model, numSteps, data, labels, device):
        '''
        Returns accuracy rate comparing to total batch size, and
        total accuracy count inside the batch size 
        '''
        with torch.no_grad():
            model.eval()
            data = data.to(device)
            labels = labels.to(device)
            spikes, _ , _ = model(data, numSteps)
            acc = SF.accuracy_rate(spikes, labels) * spikes.size(1)
            total = spikes.size(1)
        return acc/total, acc

    def fit_spike(self, model, nEpochs, opt, lossF, train_load, val_load, nSteps, device):
        '''
        Fitting function for the case when data is split to train and validation sets.

        model:        spike-network model to train
        nEpochs:      number of epochs for training
        opt:          optimizer
        lossF:        loss function
        train_load:   loader for train data
        val_load:     loader for validation data
        nSteps:       number of steps in one exitation
        device:       device to use for training (cpu or cuda)
        history:      list of statistics for all epochs (appended in each epoch)
        '''
        history = []

        for i in range(nEpochs):
            model.train()
            # Define lists to store training loss and accuracy
            tLoss = []
            tAcc = list()
            for batch, labels in train_load:
                batch = batch.to(device)
                labels = labels.to(device)
                # Generate predictions/ forward pass
                spikes, _, _ = model(batch, nSteps)
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
                a, _ = self.accuracy_spike(model, nSteps, batch, labels, device)
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
                spikes, _, _ = model(batch, nSteps)
                with torch.no_grad():
                    loss = lossF(spikes, labels)
                vLoss.append(loss.detach().item())
                a, _ = self.accuracy_spike(model, nSteps, batch, labels, device)
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
    
    def fit_spike_full_train(self, model, startEpoch, nEpochs, ResNetModel, dataSet, sched, opt, lossF, train_load, nSteps, gd, device, checkpointPeriod=1):
        '''
        Fitting function for the case when no validation set is used.

        model:        spike-network model to train
        nEpochs:      number of epochs for training
        ResNetModel:  ResNet model type (9 or 10 layers), gives the label of the model
        dataSet:      dataset name (MNIST, CIFAR10, etc.)
        sched:        scheduler
        opt:          optimizer
        lossF:        loss function
        train_load:   loader for train data
        nSteps:       number of steps in one exitation !
        gd:           gradient clipping
        device:       device to use for training (cpu or cuda)
        checkpointPeriod: period of saving model weights (default is 1)
        history:      list of statistics for all epochs (appended in each epoch)     
        '''
        history = []
        current_step = 0

        for i in range(startEpoch, nEpochs):
            model.train()
            # Define lists to store training loss and accuracy
            tLoss = []
            tAcc = list()
            # for batch, labels in train_load:
            for batch_idx, (batch, labels) in enumerate(train_load):
                batch = batch.to(device)
                labels = labels.to(device)
                # Generate predictions/ forward pass
                spikes, _, _ = model(batch, nSteps)
                # Calculate loss
                loss = lossF(spikes, labels) 
                tLoss.append(loss.detach().item())
                # Set opt grad
                opt.zero_grad()
                # Update weights
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), gd)
                # Update opt
                opt.step()
                if sched is not None:
                    sched.step()
                # Check train accuracy
                a, _ = self.accuracy_spike(model, nSteps, batch, labels, device)
                tAcc.append(a.item())
                current_step = i * len(train_load) + batch_idx
                del batch, labels

            # Training stats
            meanTA = sum(tAcc) / len(tAcc)
            meanTL = sum(tLoss) / len(tLoss)           

            # Make progress bar
            suffixArray = ' ' + 'Training loss: ' + f'{meanTL:.2f} ' + 'Training accuracy: ' + f'{meanTA:.2f} '

            self.progressBar(i + 1, nEpochs, prefix = 'Progress: ', suffix = suffixArray, length = 40, fill = '#')
            currentHistory = [meanTL, meanTA]
            history.append(currentHistory)
            if i%checkpointPeriod == 0:
                fileName = 'weights/spike/' + 'resnet' + str(ResNetModel) + dataSet + '_checkpoint_' + '.pth'
                if sched is not None:
                    print(sched.state_dict())
                    print(f"sched.last_epoch: {sched.last_epoch}")
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "lr_scheduler": sched.state_dict(),
                        "epochs": i,
                        "current_step": current_step  # Store current step
                    }
                else:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epochs": i,
                        "current_step": current_step  # Store current step
                    }
                torch.save(checkpoint, fileName)

        print('\n')
        return history
    

# Define Resnet model
#####################

# 9 layers
# Downsampling is done with MaxPool2d
class SpikeResNet9Model(BasicModel):
                                                                                # Input size
    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold):     # 28x28
        super().__init__(numberOfClasses)

        self.block1 = self.convBlock(numberOfChannels, 64)                     # 64x28x28
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')                   # 64x28x28

        self.block2 = self.convBlock(64, 128)                                  # 128x28x28
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')                   # 128x28x28
        self.maxp2 = nn.MaxPool2d(2)                                            # 128x14x14

        self.resBlock3_1 = self.convBlock(128, 128)
        self.r3_lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')
        self.resBlock3_2 = self.convBlock(128, 128)
        self.r3_lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')                # 128x14x14
        
        self.block4 = self.convBlock(128, 256)                                 # 256x14x14
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')
        self.maxp4 = nn.MaxPool2d(2)                                            # 256x7x7

        self.block5 = self.convBlock(256, 512)                                 # 512x7x7
        self.lif5 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')
        self.maxp5 = nn.MaxPool2d(2)                                            # 512x3x3

        self.resBlock6_1 = self.convBlock(512, 512)
        self.r6_lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')
        self.resBlock6_2 = self.convBlock(512, 512)
        self.r6_lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')                # 512x3x3

        self.amax7 = nn.AdaptiveMaxPool2d(1)                                    # 512x1x1
        self.flat = nn.Flatten()
        self.fc7 = nn.Linear(512, numberOfClasses)
        self.lifOut = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero', output=True)

    def convBlock(self, input, output):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, padding=1, bias=False),
                  nn.BatchNorm2d(num_features=output)]
        return nn.Sequential(*layers)
    
    def forward(self, x, numberOfSteps):
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

        feat_trace = []
        prob_trace = []
        spik_trace = []

        for _ in range(numberOfSteps):

            cur1 = self.block1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            # print(f'mem1 is {mem1}')
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

            feat_trace.append(spk7)
            prob_trace.append(mem7)
            spik_trace.append(spk_out)

        # print(f'spik_trace is {spik_trace}, prob_trace is {prob_trace}')
        return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)

    # spk7 512-D spikes
    # spk_out 10-D spikes
    # mem7 10-D memebrane voltage

# 10 layers
# Downsampling is done via convolutional kernel followed by leaky activation
class SpikeResNet10Model(BasicModel):
    
    '''
    Version of spike-ResNet10 with spike block inside downsampling block
    SNN must be in residual and downsampling block before adding
    '''
    
                                                                                    # Input size
    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold):     # 32x32
        super().__init__(numberOfClasses)

        self.block1 = self.convBlock(numberOfChannels, 64)                      # 64x32x32
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') # 64x32x32

        # The residual super-block
        # Contains blocks with skip connections (2 conv blocks)
        self.resBlock2_1 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock2_2 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')

        # Downsample block
        self.downsample3 = self.convBlock(64, 128, kernel_size=1, stride=2, padding=0)
        self.d_lif3 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')

        # The residual super-block
        # Contains two blocks with skip connections (4 conv blocks)
        self.resBlock4_1 = self.convBlock(64, 128, kernel_size=3, stride=2)     # 64x32x32
        self.r4_lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock4_2 = self.convBlock(128, 128)                             # 128x16x16
        self.r4_lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')

        # Downsample block
        self.downsample5 = self.convBlock(128, 256, kernel_size=1, stride=2, padding=0)
        self.d_lif5 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')

        # The residual super-block
        # Contains two blocks with skip connections (2 conv blocks)
        self.resBlock6_1 = self.convBlock(128, 256, kernel_size=3, stride=2)    # 128x16x16
        self.r6_lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock6_2 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')

        # Downsample block
        self.downsample7 = self.convBlock(256, 512, kernel_size=1, stride=2, padding=0)
        self.d_lif7 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')
 
        # The residual super-block
        # Contains two blocks with skip connections (2 conv blocks)
        self.resBlock8_1 = self.convBlock(256, 512, kernel_size=3, stride=2)    # 512x4x4
        self.r8_lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock8_2 = self.convBlock(512, 512)                             # 512x4x4
        self.r8_lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')

        self.amax9 = nn.AdaptiveMaxPool2d(1)                                    # 512x1x1
        self.flat = nn.Flatten()
        self.fc9 = nn.Linear(512, numberOfClasses)
        self.lifOut = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero', output=True)

    def convBlock(self, input, output, kernel_size=3, stride=1, padding=1):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(num_features=output)]
        return nn.Sequential(*layers)
    
    def forward(self, x, numberOfSteps):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2_1 = self.r2_lif1.init_leaky()
        mem2_2 = self.r2_lif2.init_leaky()
        mem4_1 = self.r4_lif1.init_leaky()
        mem4_2 = self.r4_lif2.init_leaky()
        mem6_1 = self.r6_lif1.init_leaky()
        mem6_2 = self.r6_lif2.init_leaky()
        mem8_1 = self.r8_lif1.init_leaky()
        mem8_2 = self.r8_lif2.init_leaky()
        mem9 = self.lifOut.init_leaky()
        mem_d3 = self.d_lif3.init_leaky()
        mem_d5 = self.d_lif5.init_leaky()
        mem_d7 = self.d_lif7.init_leaky()

        feat_trace = []
        prob_trace = []
        spik_trace = []

        for _ in range(numberOfSteps):

            cur1 = self.block1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2_1 = self.resBlock2_1(spk1)
            spk2_1, mem2_1 = self.r2_lif1(cur2_1, mem2_1)
            cur2_2 = self.resBlock2_2(spk2_1)
            spk2_2, mem2_2 = self.r2_lif2(cur2_2, mem2_2)
            # make skip connection, ADD
            spk_r2_1 = spk2_2 + spk1
            
            # Block input conv (64, 128)
            id_3 = self.downsample3(spk_r2_1)
            identity_3, mem_d3 = self.d_lif3(id_3, mem_d3)
            # print(f'identity_4.shape is {identity_4.shape}')
            cur4_1 = self.resBlock4_1(spk_r2_1)
            spk4_1, mem4_1 = self.r4_lif1(cur4_1, mem4_1)
            cur4_2 = self.resBlock4_2(spk4_1)
            spk4_2, mem4_2 = self.r4_lif2(cur4_2, mem4_2)
            # make skip connection
            spk_r4_1 = spk4_2 + identity_3

            # Block input conv (128, 256)
            id_5 = self.downsample5(spk_r4_1)
            identity_5, mem_d5 = self.d_lif5(id_5, mem_d5)
            # print(f'identity_6.shape is {identity_6.shape}')
            cur6_1 = self.resBlock6_1(spk_r4_1)
            spk6_1, mem6_1 = self.r6_lif1(cur6_1, mem6_1)
            cur6_2 = self.resBlock6_2(spk6_1)
            spk6_2, mem6_2 = self.r6_lif2(cur6_2, mem6_2)
            # make skip connection
            spk_r6_1 = spk6_2 + identity_5

            # Block input conv (256, 512)
            id_7 = self.downsample7(spk_r6_1)
            identity_7, mem_d7 = self.d_lif7(id_7, mem_d7)
            # print(f'identity_8.shape is {identity_8.shape}')
            cur8_1 = self.resBlock8_1(spk_r6_1)
            spk8_1, mem8_1 = self.r8_lif1(cur8_1, mem8_1)
            cur8_2 = self.resBlock8_2(spk8_1)
            spk8_2, mem8_2 = self.r8_lif2(cur8_2, mem8_2)
            # make skip connection
            spk_r8_1 = spk8_2 + identity_7

            spk9 = self.amax9(spk_r8_1)
            spk9 = self.flat(spk9)
            cur9 = self.fc9(spk9)
            spk_out, mem9 = self.lifOut(cur9, mem9)

            feat_trace.append(spk9)
            prob_trace.append(mem9)
            spik_trace.append(spk_out)

        # print(f'spik_trace is {spik_trace}, prob_trace is {prob_trace}')
        return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)

# 18 layers
class SpikeResNet18Model(BasicModel):                                           # Input size

    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold):     # 32x32
        super().__init__(numberOfClasses)

        self.block1 = self.convBlock(numberOfChannels, 64)                      # 64x32x32
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')                   # 64x32x32

        # The residual super-block
        # Contains two blocks with skip connections (4 conv blocks)
        self.resBlock2_1 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock2_2 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')
        self.resBlock2_3 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_lif3 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock2_4 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_lif4 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')  
        
        # Downsample block
        self.downsample3 = self.convBlock(64, 128, kernel_size=1, stride=2, padding=0)
        self.d_lif3 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')

        # The residual super-block
        # Contains two blocks with skip connections ( 4 conv blocks)
        self.resBlock4_1 = self.convBlock(64, 128, kernel_size=3, stride=2)     # 64x32x32
        self.r4_lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock4_2 = self.convBlock(128, 128)                             # 128x16x16
        # Here is the place for skip connection
        self.r4_lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')
        self.resBlock4_3 = self.convBlock(128, 128)                             # 128x16x16
        self.r4_lif3 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock4_4 = self.convBlock(128, 128)                             # 128x16x16
        # Here is the place for skip connection
        self.r4_lif4 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 

        # Downsample block
        self.downsample5 = self.convBlock(128, 256, kernel_size=1, stride=2, padding=0)
        self.d_lif5 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')

        # The residual super-block
        # Contains two blocks with skip connections ( 4 conv blocks)
        self.resBlock6_1 = self.convBlock(128, 256, kernel_size=3, stride=2)    # 128x16x16
        self.r6_lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock6_2 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')
        self.resBlock6_3 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_lif3 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock6_4 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_lif4 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 

        # Downsample block
        self.downsample7 = self.convBlock(256, 512, kernel_size=1, stride=2, padding=0)
        self.d_lif7 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')

        # The residual super-block
        # Contains two blocks with skip connections ( 4 conv blocks)
        self.resBlock8_1 = self.convBlock(256, 512, kernel_size=3, stride=2)    # 256x8x8
        self.r8_lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock8_2 = self.convBlock(512, 512)                             # 512x4x4
        # Here is the place for skip connection
        self.r8_lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero')
        self.resBlock8_3 = self.convBlock(512, 512)                             # 512x4x4
        self.r8_lif3 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 
        self.resBlock8_4 = self.convBlock(512, 512)                             # 512x4x4
        # Here is the place for skip connection
        self.r8_lif4 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero') 

        self.amax9 = nn.AdaptiveMaxPool2d(1)
        self.flat = nn.Flatten()
        self.fc10 = nn.Linear(512, numberOfClasses)
        self.lifOut = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='none', output=True)

    def convBlock(self, input, output, kernel_size=3, stride=1, padding=1):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(num_features=output)]
        return nn.Sequential(*layers)
    
    def forward(self, x, numberOfSteps):

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
        mem_d3 = self.d_lif3.init_leaky()
        mem_d5 = self.d_lif5.init_leaky()
        mem_d7 = self.d_lif7.init_leaky()

        feat_trace = []
        prob_trace = []
        spik_trace = []

        for _ in range(numberOfSteps):

            cur1 = self.block1(x)                       # Conv block gives current for the following 
            spk1, mem1 = self.lif1(cur1, mem1)          # Leaky gives spk1 and membrane voltage
            # spk1 is used for skip connection

            # Block input conv (64, 64)
            cur2_1 = self.resBlock2_1(spk1)
            spk2_1, mem2_1 = self.r2_lif1(cur2_1, mem2_1)
            cur2_2 = self.resBlock2_2(spk2_1)
            spk2_2, mem2_2 = self.r2_lif2(cur2_2, mem2_2)
            # make skip connection, with the same size
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
            id_3 = self.downsample3(spk_r2_2)
            identity_3, mem_d3 = self.d_lif3(id_3, mem_d3)
            # print(f'identity_4.shape is {identity_4.shape}')
            cur4_1 = self.resBlock4_1(spk_r2_2)
            spk4_1, mem4_1 = self.r4_lif1(cur4_1, mem4_1)
            cur4_2 = self.resBlock4_2(spk4_1)
            spk4_2, mem4_2 = self.r4_lif2(cur4_2, mem4_2)
            # make skip connection
            spk_r4_1 = spk4_2 + identity_3
            cur4_3 = self.resBlock4_3(spk_r4_1)
            spk4_3, mem4_3 = self.r4_lif3(cur4_3, mem4_3)
            cur4_4 = self.resBlock4_4(spk4_3)
            spk4_4, mem4_4 = self.r4_lif4(cur4_4, mem4_4)
            # make skip connection
            spk_r4_2 = spk4_4 + spk_r4_1

            # Block input conv (128, 256)
            id_5 = self.downsample5(spk_r4_2)
            identity_5, mem_d5 = self.d_lif5(id_5, mem_d5)
            cur6_1 = self.resBlock6_1(spk_r4_2)
            spk6_1, mem6_1 = self.r6_lif1(cur6_1, mem6_1)
            cur6_2 = self.resBlock6_2(spk6_1)
            spk6_2, mem6_2 = self.r6_lif2(cur6_2, mem6_2)
            # make skip connection
            spk_r6_1 = spk6_2 + identity_5
            cur6_3 = self.resBlock6_3(spk_r6_1)
            spk6_3, mem6_3 = self.r6_lif3(cur6_3, mem6_3)
            cur6_4 = self.resBlock6_4(spk6_3)
            spk6_4, mem6_4 = self.r6_lif4(cur6_4, mem6_4)
            # make skip connection
            spk_r6_2 = spk6_4 + spk_r6_1

            # Block input conv (256, 512)
            id_7 = self.downsample7(spk_r6_2)
            identity_7, mem_d7 = self.d_lif7(id_7, mem_d7)
            cur8_1 = self.resBlock8_1(spk_r6_2)
            spk8_1, mem8_1 = self.r8_lif1(cur8_1, mem8_1)
            cur8_2 = self.resBlock8_2(spk8_1)
            spk8_2, mem8_2 = self.r8_lif2(cur8_2, mem8_2)
            # make skip connection
            spk_r8_1 = spk8_2 + identity_7
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

            feat_trace.append(spk9)
            prob_trace.append(mem_out)
            spik_trace.append(spk_out)

        return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)

# Good for grayscale
class spikeConvNN1(BasicModel):
    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold):
        # it was set to init threshold value 0.2
        super().__init__(numberOfClasses)

        self.conv1 = nn.Conv2d(numberOfChannels, 128, kernel_size=3, padding=1, stride=2)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
        self.amax = nn.AdaptiveMaxPool2d(1)
        self.flat = nn.Flatten()    
        self.fc4 = nn.Linear(512, numberOfClasses)
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
            print(f"Conv1 x.shape: {x.shape}")
            cur1 = self.conv1(x)
            print(f"Conv1 cur1.shape: {cur1.shape}")
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.conv2(spk1)
            print(f"Conv1 cur2.shape: {cur2.shape}")
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.conv3(spk2)
            print(f"Conv1 cur3.shape: {cur3.shape}")
            spk3, mem3 = self.lif3(cur3, mem3)
 
            print(f"Conv1 spk3.shape: {spk3.shape}")
            spk4 = self.amax(spk3)
            print(f"Conv1 spk4.shape: {spk4.shape}")
            spk4 = self.flat(spk4)
            cur4 = self.fc4(spk4)
            spk_out, mem4 = self.lif4(cur4, mem4)

            feat_trace.append(spk4)
            prob_trace.append(mem4)
            spik_trace.append(spk_out)

        return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)

# spike convolutional network
# ResNetModel2
class spikeConvNN2(BasicModel):
    '''
    Convolutional network model from:
    https://github.com/aitor-martinez-seras/OoD_on_SNNs/blob/main/Explainable_OoD_detection_on_SNNs.ipynb
    https://arxiv.org/abs/2210.00894

    This is the convolutional model with two hidden layers
    threshold should be set to 0.2

    check the model in Norse.LIFCell alpha=100 (what does it mean)?
    let's hold to the SNN.Leaky with beta decay

    '''
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

    def forward(self, x, num_steps):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        
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


class spikeConvNN3(BasicModel):
    def __init__(self, numberOfChannels, numberOfClasses, beta, threshold):
        # it was set to init threshold value 0.2
        super().__init__(numberOfClasses)

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
            spk_out, mem4 = self.lif4(cur4, mem4)

            feat_trace.append(spk4)
            prob_trace.append(mem4)
            spik_trace.append(spk_out)

        return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)


# old
# class spikeConvNN1(BasicModel):
#     def __init__(self, numberOfChannels, numberOfClasses, beta, threshold, feature_size=28):
#         # it was set to init threshold value 0.2
#         super().__init__(numberOfClasses)

#         # self.features = int(((feature_size -2)/2)-2)
#         # self.averaging = int((feature_size - 2)/2)

#         # self.conv1 = nn.Conv2d(numberOfChannels, 16, kernel_size=3)
#         # self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
#         # self.max1 = nn.AdaptiveMaxPool2d((self.averaging))
#         # self.conv2 = nn.Conv2d(16, 64, kernel_size=3)
#         # self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
#         # self.fc3 = nn.Linear(self.features * self.features * 64, 512)
#         # self.lif3 = snn.Leaky(beta=beta, threshold=threshold/2, reset_mechanism="zero") 
#         # self.fc4 = nn.Linear(512, 256)
#         # self.lif4 = snn.Leaky(beta=beta, threshold=threshold/4, reset_mechanism='zero')  
#         # self.fc5 = nn.Linear(256, numberOfClasses)
#         # self.lif5 = snn.Leaky(beta=beta)

#         self.conv1 = nn.Conv2d(numberOfChannels, 64, kernel_size=3, padding=1, stride=2, bias=False)
#         self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False)
#         self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False)
#         self.lif3 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
#         self.aavg = nn.AdaptiveAvgPool2d(1)
#         self.flat = nn.Flatten()    
#         self.fc4 = nn.Linear(256, numberOfClasses, bias=False)
#         self.lif4 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='zero', output=True)

#     def forward(self, x, num_steps):

#         # Initialize hidden states and outputs at t=0
#         mem1 = self.lif1.init_leaky()
#         mem2 = self.lif2.init_leaky()
#         mem3 = self.lif3.init_leaky()
#         mem4 = self.lif4.init_leaky()
#         # mem5 = self.lif5.init_leaky()
        
#         feat_trace = []
#         prob_trace = []
#         spik_trace = []

#         for _ in range(num_steps):
#             cur1 = self.conv1(x)
#             spk1, mem1 = self.lif1(cur1, mem1)
#             # spk1 = self.max1(spk1)

#             cur2 = self.conv2(spk1)
#             spk2, mem2 = self.lif2(cur2, mem2)
#             # spk2 = spk2.view(-1, self.features * self.features * 64)

#             cur3 = self.conv3(spk2)
#             spk3, mem3 = self.lif3(cur3, mem3)

#             spk3 = self.aavg(spk3)
#             spk3 = spk3.view(-1, 256)
#             cur4 = self.fc4(spk3)
#             spk4, mem4 = self.lif4(cur4, mem4)

#             # cur5 = self.fc5(spk4)
#             # spk_out, mem5 = self.lif5(cur5, mem5)

#             feat_trace.append(spk3)
#             prob_trace.append(mem4)
#             spik_trace.append(spk4)

#         return torch.stack(spik_trace, dim=0), torch.stack(feat_trace, dim=0), torch.stack(prob_trace, dim=0)

