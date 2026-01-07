''' 
Author: Aleksej Avramovic
Last update: 27/08/2024

The first step towards spiking In Distribution / Out of Distribution detection

Resnet9 for MNIST
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
sys.path.append('../')


# Define basic model
####################

class BasicModel(nn.Module):

    def __init__(self, nClasses):
        super().__init__()
        self.numberOfClasses = nClasses

    # TODO: Make test for this
    # TODO: Extend this function for multiple inputs
    # NOTE: Maybe not necessary (to be removed)
    def softmax(self, input):
        e = torch.exp(input)
        p = e / torch.sum(e)
        return p 
    
    def progressBar(self, iter, total, prefix = '', suffix = '', length = 30, fill = '#'):
        percent = f'{100 * (iter / (float(total))):.1f}'
        filled = int(length * iter // total)
        bar = fill * filled + '_' * (length - filled) + ' ' + percent
        sys.stdout.write('\r%s |%s%% %s' % (prefix, bar, suffix))
        sys.stdout.flush()
        return None
    
    def accuracy(self, pred, truth):
        _, o = torch.max(pred, dim = 1)
        return torch.tensor(torch.sum(o == truth).item() / len(truth))
    
    # TODO: Define train and valid steps separately 
    # also validation needs to have decorator @torch.no_grad()
    
    def fit_conv(self, nEpochs, model, lossFunction, lr, train_load, val_load, history, device, wd = 0, gd = None):
        '''
        nEpochs:      number of epochs for training
        model:        architecure of network
        lossFunction: loss function
        lr:           learning rate
        wd:           weight decay
        gd:           gradient clipping
        train_load:   loader for train data
        val_load:     loader for validation data
        history:      history of the statistics of previous batches
        device:       CPU or GPU
        '''

        # Define optimization
        opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)
        # opt = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=nEpochs, steps_per_epoch=len(train_load))
        for i in range(nEpochs):
            # Training
            model.train()
            # Define lists to store training loss and accuracy
            tLoss = []
            tAcc = list()
            for batch, labels in train_load:
                batch = batch.to(device)
                labels = labels.to(device)
                # Generate predictions
                pred, _ = model(batch)
                # Calculate loss
                loss = lossFunction(pred, labels)
                tLoss.append(loss.detach().item())
                # Calculate gradients
                loss.backward()
                if gd:
                    nn.utils.clip_grad_value_(model.parameters(), gd)
                # Update parameters
                opt.step()
                sched.step()
                # Reset gradiensts
                opt.zero_grad()
                # Check train accuracy
                a = self.accuracy(pred, labels)
                tAcc.append(a.item())

                del batch, labels, pred, loss
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
                pred, _ = model(batch)
                with torch.no_grad():
                    loss = lossFunction(pred, labels)
                vLoss.append(loss.detach().item())
                a = self.accuracy(pred, labels)
                vAcc.append(a.item())
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
        return model, history 
    
    def fit_conv_full_train(self, startEpoch, nEpochs, model, lossFunction, train_load, ResNetModel, dataSet, sched, opt, gd, device, checkpointFile, checkpointPeriod=1):
        '''
        Updated to align with fit_membrane_full_train.
        Includes checkpointing, execution time tracking, and handling of model.expansion > 1.
        '''
        history = []
        current_step = 0

        for i in range(startEpoch, nEpochs):
            model.train()
            tLoss = []
            total_correct = 0
            total_samples = 0
            start_time = time.time()

            for batch_idx, (batch, labels) in enumerate(train_load):
                batch = batch.to(device)
                labels = labels.to(device)

                # Forward pass
                pred, _ = model(batch)

                # Handle population coding if model.expan > 1
                if hasattr(model, 'expan') and model.expan > 1:
                    pred = pred.reshape(pred.shape[0], model.numberOfClasses, model.expan).sum(dim=2)

                # Calculate loss
                loss = lossFunction(pred, labels)
                tLoss.append(loss.detach().item())

                # Backward pass and optimization
                opt.zero_grad()
                loss.backward()
                if gd:
                    nn.utils.clip_grad_value_(model.parameters(), gd)
                opt.step()
                if sched is not None:
                    sched.step()

                # Accuracy calculation
                with torch.no_grad():
                    predicted = torch.argmax(pred, dim=1)
                    correct = (predicted == labels).float().sum()
                    total_correct += correct.item()
                    total_samples += labels.size(0)

                current_step = i * len(train_load) + batch_idx
                del batch, labels

            # Epoch stats
            end_time = time.time()
            execution_time = end_time - start_time
            meanTA = total_correct / total_samples
            meanTL = sum(tLoss) / len(tLoss)

            # Progress bar
            suffixArray = f' Training loss: {meanTL:.2f} Training accuracy: {meanTA:.2f} Time: {execution_time:.2f}s'
            self.progressBar(i + 1, nEpochs, prefix='Progress: ', suffix=suffixArray, length=40, fill='#')
            history.append([meanTL, meanTA])

            # Checkpoint saving
            if i % checkpointPeriod == 0:
                fileName = checkpointFile
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "lr_scheduler": sched.state_dict() if sched is not None else None,
                    "epochs": i,
                    "current_step": current_step
                }
                torch.save(checkpoint, fileName)

        print('\n')
        return history

# Define Resnet model
#####################

class convNN4(BasicModel):
    '''
    Let's modify an architecture a little bit. 
    Add population coding (large number of output neurons).

    For BPTT implementation (3rd party, we must implement only one time step)
    Ours can go further
    '''

    def __init__(self, numberOfChannels, numberOfClasses, feature_size=28, expan=1):

        super().__init__(numberOfClasses)
        self.numberOfClasses = numberOfClasses
        self.expan = expan 

        self.features = int(((feature_size - 2) / 2) - 2)
        self.averaging = int((feature_size - 2) / 2)

        self.conv1 = nn.Conv2d(numberOfChannels, 32, kernel_size=3, bias=False)
        self.lif1 = nn.ReLU() 
        self.avg1 = nn.AdaptiveAvgPool2d(self.averaging)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.lif2 = nn.ReLU() 
        self.fc3 = nn.Linear(self.features * self.features * 64, 512, bias=False)
        self.lif3 = nn.ReLU() 
        self.fc4 = nn.Linear(512, 256, bias=False)
        self.lif4 = nn.ReLU() 
        self.fc5 = nn.Linear(256, numberOfClasses * self.expan, bias=False)
        self.lif5 = nn.ReLU() 

    def forward(self, x):
        # Pass through the first convolutional layer and activation
        x = self.conv1(x)
        x = self.lif1(x)
        x = self.avg1(x)

        # Pass through the second convolutional layer and activation
        x = self.conv2(x)
        x = self.lif2(x)

        # Flatten the output for the fully connected layers
        x = x.view(-1, self.features * self.features * 64)

        # Pass through the fully connected layers with activations
        x = self.fc3(x)
        x = self.lif3(x)

        x = self.fc4(x)
        x = self.lif4(x)

        # Final fully connected layer for predictions
        f = x  # Intermediate features
        p = self.fc5(x)
        p = self.lif5(p)

        return p, f


class ResNet9Model(BasicModel):
    '''
    ResNet model for input images of the size 28 x 28
    '''

    def __init__(self, numberOfChannels, numberOfClasses):
        super().__init__(numberOfClasses)

        self.block1 = self.convBlock1(numberOfChannels, 64)
        self.block2 = self.convBlock2(64, 128)
        self.resBlock1 = nn.Sequential(self.convBlock1(128, 128), self.convBlock1(128, 128))

        self.block3 = self.convBlock2(128, 256)
        self.block4 = self.convBlock2(256, 512)
        self.resBlock2 = nn.Sequential(self.convBlock1(512, 512), self.convBlock1(512, 512))

        self.mpool = nn.MaxPool2d(3)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512, numberOfClasses)

    def convBlock1(self, input, output):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, padding=1),
                  nn.BatchNorm2d(num_features=output),
                  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def convBlock2(self, input, output):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, padding=1),
                  nn.BatchNorm2d(num_features=output),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        return nn.Sequential(*layers)
    
    def flatLayer(self, numOfClasses):
        layers = [nn.AdaptiveMaxPool2d(1),      
                #   nn.MaxPool2d(3),  #32 za 256
                  nn.Flatten(),
                  nn.Linear(512, numOfClasses)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.resBlock1(x) + x 
        x = self.block3(x)
        x = self.block4(x)
        x = self.resBlock2(x) + x
        x = self.mpool(x)
        f = self.flat(x)
        p = self.fc(f)
        return p, f


class ResNet10(BasicModel):

    def __init__(self, numberOfChannels, numberOfClasses=10, expan = 1):
        super().__init__(numberOfClasses)
        self.in_channels = numberOfChannels
        self.expan = expan
        self.num_classes = numberOfClasses

        # Initial conv
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 1, stride=1)
        self.layer2 = self._make_layer(64, 128, 1, stride=2)
        self.layer3 = self._make_layer(128, 256, 1, stride=2)
        self.layer4 = self._make_layer(256, 512, 1, stride=2)

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes * self.expan)

    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        for i in range(blocks):
            stride_i = stride if i == 0 else 1
            layers.append(self.BasicBlock(in_channels, out_channels, stride_i))
            in_channels = out_channels  # Update in_channels after each block
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"After conv1 shape: {x.shape}")
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        return self.fc(f), f
    

class ResNet18(BasicModel):

    def __init__(self, numberOfChannels, numberOfClasses=10, expan = 1):
        super().__init__(numberOfClasses)
        self.in_channels = numberOfChannels
        self.expan = expan
        self.num_classes = numberOfClasses

        # Initial conv
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes * self.expan)

    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        for i in range(blocks):
            stride_i = stride if i == 0 else 1
            layers.append(self.BasicBlock(in_channels, out_channels, stride_i))
            in_channels = out_channels  # Update in_channels after each block
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        return self.fc(f), f
    

class newResNet10Model(BasicModel):
    '''
    Version of spike-ResNet10 with spike block inside downsampling block
    Modified to use ReLU activations instead of Leaky activations.
    Preserves input-output sizes and downsampling approach for all residual blocks.
    '''

    def __init__(self, numberOfChannels, numberOfClasses, numberOfSteps=1, expansion=1):
        super().__init__(numberOfClasses)
        self.numberOfClasses = numberOfClasses
        self.expansion = expansion
        self.numberOfSteps = numberOfSteps

        self.block1 = self.convBlock(numberOfChannels, 64)                      # 64x32x32
        self.relu1 = nn.ReLU()

        self.resBlock2_1 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_relu1 = nn.ReLU()
        self.resBlock2_2 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_relu2 = nn.ReLU()

        self.downsample3 = self.convBlock(64, 128, kernel_size=1, stride=2, padding=0)
        self.d_relu3 = nn.ReLU()

        self.resBlock4_1 = self.convBlock(64, 128, kernel_size=3, stride=2)     # 64x32x32 -> 128x16x16
        self.r4_relu1 = nn.ReLU()
        self.resBlock4_2 = self.convBlock(128, 128)                             # 128x16x16
        self.r4_relu2 = nn.ReLU()

        self.downsample5 = self.convBlock(128, 256, kernel_size=1, stride=2, padding=0)
        self.d_relu5 = nn.ReLU()

        self.resBlock6_1 = self.convBlock(128, 256, kernel_size=3, stride=2)    # 128x16x16 -> 256x8x8
        self.r6_relu1 = nn.ReLU()
        self.resBlock6_2 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_relu2 = nn.ReLU()

        self.downsample7 = self.convBlock(256, 512, kernel_size=1, stride=2, padding=0)
        self.d_relu7 = nn.ReLU()

        self.resBlock8_1 = self.convBlock(256, 512, kernel_size=3, stride=2)    # 256x8x8 -> 512x4x4
        self.r8_relu1 = nn.ReLU()
        self.resBlock8_2 = self.convBlock(512, 512)                             # 512x4x4
        self.r8_relu2 = nn.ReLU()

        self.amax9 = nn.AdaptiveMaxPool2d(1)                                    # 512x1x1
        self.flat = nn.Flatten()
        self.fc9 = nn.Linear(512, numberOfClasses * self.expansion)  # 512x1x1 -> 10x1x1

    def convBlock(self, input, output, kernel_size=3, stride=1, padding=1):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                  nn.BatchNorm2d(num_features=output)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Block 1
        x = self.block1(x)
        x = self.relu1(x)

        # Residual Block 2
        identity = x
        x = self.resBlock2_1(x)
        x = self.r2_relu1(x)
        x = self.resBlock2_2(x)
        x = self.r2_relu2(x)
        x += identity

        # Downsample Block 3
        identity = self.downsample3(x)  # Downsample the identity
        x = self.downsample3(x)
        x = self.d_relu3(x)

        # Residual Block 4
        x = self.resBlock4_1(x)
        x = self.r4_relu1(x)
        x = self.resBlock4_2(x)
        x = self.r4_relu2(x)
        x += identity

        # Downsample Block 5
        identity = self.downsample5(x)  # Downsample the identity
        x = self.downsample5(x)
        x = self.d_relu5(x)

        # Residual Block 6
        x = self.resBlock6_1(x)
        x = self.r6_relu1(x)
        x = self.resBlock6_2(x)
        x = self.r6_relu2(x)
        x += identity

        # Downsample Block 7
        identity = self.downsample7(x)  # Downsample the identity
        x = self.downsample7(x)
        x = self.d_relu7(x)

        # Residual Block 8
        x = self.resBlock8_1(x)
        x = self.r8_relu1(x)
        x = self.resBlock8_2(x)
        x = self.r8_relu2(x)
        x += identity

        # Final Layers
        x = self.amax9(x)
        f = self.flat(x)
        p = self.fc9(f)

        return p, f


class newResNet18Model(BasicModel):  
    
    '''
    Version of spike-ResNet18 with spike block inside downsampling block
    Modified to use ReLU activations instead of Leaky activations.
    Preserves multiple residual blocks per stage and ensures input-output sizes and downsampling remain unchanged.
    '''

    def __init__(self, numberOfChannels, numberOfClasses, numberOfSteps=1, expansion=1):
        super().__init__(numberOfClasses)
        self.numberOfClasses = numberOfClasses
        self.expansion = expansion
        self.num_steps = numberOfSteps

        self.block1 = self.convBlock(numberOfChannels, 64)                      # 64x32x32
        self.relu1 = nn.ReLU()

        self.resBlock2_1 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_relu1 = nn.ReLU()
        self.resBlock2_2 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_relu2 = nn.ReLU()
        self.resBlock2_3 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_relu3 = nn.ReLU()
        self.resBlock2_4 = self.convBlock(64, 64)                               # 64x32x32
        self.r2_relu4 = nn.ReLU()

        self.downsample3 = self.convBlock(64, 128, kernel_size=1, stride=2, padding=0)
        self.d_relu3 = nn.ReLU()

        self.resBlock4_1 = self.convBlock(64, 128, kernel_size=3, stride=2)     # 64x32x32 -> 128x16x16
        self.r4_relu1 = nn.ReLU()
        self.resBlock4_2 = self.convBlock(128, 128)                             # 128x16x16
        self.r4_relu2 = nn.ReLU()
        self.resBlock4_3 = self.convBlock(128, 128)                             # 128x16x16
        self.r4_relu3 = nn.ReLU()
        self.resBlock4_4 = self.convBlock(128, 128)                             # 128x16x16
        self.r4_relu4 = nn.ReLU()

        self.downsample5 = self.convBlock(128, 256, kernel_size=1, stride=2, padding=0)
        self.d_relu5 = nn.ReLU()

        self.resBlock6_1 = self.convBlock(128, 256, kernel_size=3, stride=2)    # 128x16x16 -> 256x8x8
        self.r6_relu1 = nn.ReLU()
        self.resBlock6_2 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_relu2 = nn.ReLU()
        self.resBlock6_3 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_relu3 = nn.ReLU()
        self.resBlock6_4 = self.convBlock(256, 256)                             # 256x8x8
        self.r6_relu4 = nn.ReLU()

        self.downsample7 = self.convBlock(256, 512, kernel_size=1, stride=2, padding=0)
        self.d_relu7 = nn.ReLU()

        self.resBlock8_1 = self.convBlock(256, 512, kernel_size=3, stride=2)    # 256x8x8 -> 512x4x4
        self.r8_relu1 = nn.ReLU()
        self.resBlock8_2 = self.convBlock(512, 512)                             # 512x4x4
        self.r8_relu2 = nn.ReLU()
        self.resBlock8_3 = self.convBlock(512, 512)                             # 512x4x4
        self.r8_relu3 = nn.ReLU()
        self.resBlock8_4 = self.convBlock(512, 512)                             # 512x4x4
        self.r8_relu4 = nn.ReLU()

        self.amax9 = nn.AdaptiveMaxPool2d(1)                                    # 512x1x1
        self.flat = nn.Flatten()
        self.fc9 = nn.Linear(512, numberOfClasses * self.expansion)  # 512x1x1 -> 10x1x1

    def convBlock(self, input, output, kernel_size=3, stride=1, padding=1):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                  nn.BatchNorm2d(num_features=output)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Block 1
        x = self.block1(x)
        x = self.relu1(x)

        # Residual Block 2
        identity = x
        x = self.resBlock2_1(x)
        x = self.r2_relu1(x)
        x = self.resBlock2_2(x)
        x = self.r2_relu2(x)
        x = self.resBlock2_3(x)
        x = self.r2_relu3(x)
        x = self.resBlock2_4(x)
        x = self.r2_relu4(x)
        x += identity

        # Downsample Block 3
        identity = self.downsample3(x)  # Downsample the identity
        x = self.downsample3(x)
        x = self.d_relu3(x)

        # Residual Block 4
        x = self.resBlock4_1(x)
        x = self.r4_relu1(x)
        x = self.resBlock4_2(x)
        x = self.r4_relu2(x)
        x = self.resBlock4_3(x)
        x = self.r4_relu3(x)
        x = self.resBlock4_4(x)
        x = self.r4_relu4(x)
        x += identity

        # Downsample Block 5
        identity = self.downsample5(x)  # Downsample the identity
        x = self.downsample5(x)
        x = self.d_relu5(x)

        # Residual Block 6
        x = self.resBlock6_1(x)
        x = self.r6_relu1(x)
        x = self.resBlock6_2(x)
        x = self.r6_relu2(x)
        x = self.resBlock6_3(x)
        x = self.r6_relu3(x)
        x = self.resBlock6_4(x)
        x = self.r6_relu4(x)
        x += identity

        # Downsample Block 7
        identity = self.downsample7(x)  # Downsample the identity
        x = self.downsample7(x)
        x = self.d_relu7(x)

        # Residual Block 8
        x = self.resBlock8_1(x)
        x = self.r8_relu1(x)
        x = self.resBlock8_2(x)
        x = self.r8_relu2(x)
        x = self.resBlock8_3(x)
        x = self.r8_relu3(x)
        x = self.resBlock8_4(x)
        x = self.r8_relu4(x)
        x += identity

        # Final Layers
        x = self.amax9(x)
        f = self.flat(x)
        p = self.fc9(f)

        return p, f
