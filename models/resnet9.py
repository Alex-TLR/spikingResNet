''' 
Author: Aleksej Avramovic
Last update: 20/07/2024

The first step towards Out of Distribution detection

Resnet9 for MNIST
'''

import torch 
import torch.nn as nn
import sys
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
    
    def fit(self, nEpochs, model, lossFunction, lr, train_load, val_load, history, device, wd = 0, gd = None):
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
    
    def fit_full_train(self, nEpochs, model, lossFunction, lr, train_load, history, device, wd = 0, gd = None):
    # def fit(self, nEpochs, model, lossFunction, train_load, val_load, history, gd = None):
        '''
        nEpochs:      number of epochs for training
        model:        architecure of network
        lossFunction: loss function
        lr:           learning rate
        wd:           weight decay
        gd:           gradient clipping
        train_load:   loader for train data
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

            # Make progress bar
            suffixArray = ' ' + 'Training loss: ' + f'{meanTL:.2f} ' + 'Training accuracy: ' + f'{meanTA:.2f} ' 

            self.progressBar(i + 1, nEpochs, prefix = 'Progress: ', suffix = suffixArray, length = 40, fill = '#')
            currentHistory = [meanTL, meanTA]
            history.append(currentHistory)
        
        print('\n')
        return model, history 

# Define Resnet model
#####################

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

        # self.classifier = self.flatLayer(numberOfClasses)
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
        layers = [
                  nn.AdaptiveMaxPool2d(1),      
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

# class FeatureExtractorConv(nn.Module):
#     '''
#     Feature extraction for convolutional ResNet9 network
#     '''

#     def __init__(self, network):
#         super().__init__()
#         self.network = network 
#         del self.network.fc

#     def forward(self, x):
#         z = self.network(x)
#         return z
    
# class ClassifierConv(nn.Module):
#     '''
#     Classifier based on features from convolutional ResNet9
#     '''
#     def __init__(self, numberOfClasses):
#         super().__init__()
#         self.fc = nn.Linear(512, numberOfClasses)

#     def forward(self, x):
#         return self.fc(x)

# # Combine feature extractor and classifier
# class CustomResNetConv(nn.Module):
#     def __init__(self, num_classes, network):
#         super(CustomResNetConv, self).__init__()
#         self.feature_extractor = FeatureExtractorConv(network)
#         self.custom_classifier = ClassifierConv(num_classes)
#     def forward(self, x):
#         x = self.feature_extractor(x)
#         y = self.custom_classifier(x)
#         return (x, y)
    
