from utils.Utils import Utils
#from metrics.Metrics import Metrics
from torch.utils.data import random_split
import torch.nn as nn
import torch 
import matplotlib.pyplot as plt
from torchsummary import summary
from models.resnet import ResNet9Model 
from models.spikeresnet import spikeConvNN1, spikeConvNN2,  SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model  
from models.plain import spikeLinearNet1
import snntorch.functional as SF
import numpy as np
from snntorch import utils
from snntorch import spikegen
import sys
import time
import traceback
import math
from utils.Utils import CIFAR10Policy

from spikingjelly.clock_driven import neuron, surrogate, functional
from spikingjelly.clock_driven.model import sew_resnet
from torch.cuda import amp

_seed_ = 1984
import random
import math
random.seed(_seed_)
np.random.seed(_seed_)

torch.manual_seed(_seed_)  
torch.cuda.manual_seed_all(_seed_) 
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False 
import os
os.environ['PYTHONHASHSEED'] = str(_seed_)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def training(dataSet, modelType, batchSize, numOfClasses, ResNetModel, epochs=200, fullTrain=False, auto_aug=False, pretrained=False):
    '''
    dataSet:        defines the data set for training (for example MNIST, FMNIST, KMNIST)
    modelType:      convolutional or spiking neural network
    case:           case needs to contain the details of the case scenario
    fullTrain:      define if training is done on complete training set or train/valid split is used
    ResNetModel:    determines number of layers in model
    '''

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Load datase
    dataset_train, dataset_test = Utils.load_data(dataSet, auto_aug=auto_aug)

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, dataSet)
    # print(f"Image size: {channels, rows, cols}")
    sys.stdout.flush() 

    # Define batch size
    batchSize = batchSize
    
    if (fullTrain == False):
        train_loader, val_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, False, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(_seed_))
    else:
        train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(_seed_))

    # Get the device
    device = Utils.get_device()

    # Define training parameters
    # Number of classes
    numberOfClasses = numOfClasses

    numberOfEpochs = epochs
    # Number of channels
    numberOfChannels = channels

    # Gradient clipping 
    gClip = 0.5
    # Typical ranges
    # gClip = 0.5   # Common choice
    # gClip = 1.0   # Standard for many SNNs
    # gClip = 2.0   # For very deep networks

    # Weight decay
    wDecay = 0.0001

    if modelType == 'conv':    

        # Learning rate (KEY)
        lr = [0.0001]

        # Number of epochs
        numberOfEpochs = [20]

        # Loss function
        lossFunction = nn.CrossEntropyLoss()

        # Model
        if ResNetModel == 9:
            model = ResNet9Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses)
        else:
            print("Not defined")
            return -1
        
        # Move model to device
        model.to(device)

        summary(model, (3, 32, 32))

        # Keeps accuracy and loss for both training and validation in each epoch
        H = []

        # Training
        ##########

        for i in range(len(lr)):
            lrCurrent = lr[i]
            nEpochs = numberOfEpochs[i]
            if (fullTrain == False):
                model, H = model.fit_conv(nEpochs, model, lossFunction, lrCurrent, train_loader, val_loader, H, device, wd=wDecay, gd=gClip)
            elif (fullTrain == True):
                model, H = model.fit_conv_full_train(nEpochs, model, lossFunction, lrCurrent, train_loader, H, device, wd=wDecay, gd=gClip)
        print("\n")

        weightPath = 'weights/conv/resnet9_weights_' + dataSet + '.pth'
        torch.save(model.state_dict(), weightPath)

        if (fullTrain == False):
            # Plot loss for train/validation data
            tLoss = [v[0] for v in H]
            vLoss = [v[2] for v in H]

            plt.figure(figsize = (5, 5))
            plt.plot(tLoss, '-bx')
            plt.plot(vLoss, '-rx')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(['Training', 'Validation'])
            plt.title('Loss/epochs')
            plt.show()

        print("Check test images.")
        testAcc = []
        testLoss = []
        for batch, labels in test_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            output, _ = model(batch)
            l = lossFunction(output, labels)
            testLoss.append(l.item())
            a = model.accuracy(output, labels)
            testAcc.append(a.item())
        # Test stats
        meanA = sum(testAcc) / len(testAcc)
        meanL = sum(testLoss) / len(testLoss)
        print(f'Test loss is {meanL:.2f}. Test accuracy is {meanA:.2f}.')

        return None

    elif modelType == 'spike':

        # Keeps accuracy and loss for both training and validation in each epoch
        H = []

        # For spiking neural network we need number of steps
        numberOfSteps = 4
        beta = 0.95
        threshold = 0.25

        # Define model
        if ResNetModel == 1:
            model = spikeConvNN1(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 2:
            model = spikeConvNN2(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
            model = model.to(device)
            # model.reset_mem(batchSize, device)
        elif ResNetModel == 9:
            model = SpikeResNet9Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 10:
            model = SpikeResNet10Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
            # model.reset_mem(batchSize, device)
            print(model)
            model = model.to(device)
        elif ResNetModel == 18:
            model = SpikeResNet18Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold, numberOfSteps=numberOfSteps)
            print(model)
            model = model.to(device)
        elif ResNetModel == 21:
            model = spikeLinearNet1(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 22:
            T = 4
            backend = 'torch'
            model = sew_resnet.multi_step_sew_resnet18(pretrained=False, progress=True, T=T, cnf='ADD', multi_step_neuron=neuron.MultiStepIFNode, v_threshold=1., surrogate_function=surrogate.ATan(), detach_reset=True, backend=backend, num_classes=10)
            model = model.to(device)
        else:
            print("Not defined")
            return -1
        
        if ResNetModel != 22:

            # For all my models except SEW ResNet

            # Loss function
            # loss_fn = SF.ce_rate_loss()
            # loss_fn = SF.ce_count_loss() 
            loss_fn = nn.CrossEntropyLoss()
            # Optimizer for gray 5e-4
            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=wDecay)
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=wDecay)
            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=epochs, steps_per_epoch=len(train_loader))
            # sched = None
            # Training
            startEpoch = 0
            if (fullTrain == True):
                print(pretrained)
                if(pretrained == True):
                    weightsName = 'weights/spike/' + 'resnet' + str(ResNetModel) + dataSet + '_checkpoint_' + '.pth'
                    print("Try to load checkpoint: "+weightsName)
                    try:
                        file = torch.load(weightsName)
                        model.load_state_dict(file["model"])
                        optimizer.load_state_dict(file["optimizer"])
                        sched.load_state_dict(file["lr_scheduler"])
                        startEpoch=file["epochs"] + 1
                        print(f"numberOfEpochs: {numberOfEpochs}, startEpoch: {startEpoch}, steps_per_epoch: {len(train_loader)}, total_steps: {sched.total_steps}")
                        #sched._step_count = startEpoch * len(train_loader)
                        print("Checkpoint loaded.")
                    except Exception:
                        traceback.print_exc()
                        print("No valid checkpoint found. Starting from scratch.")
                    print("Training started")
                    sys.stdout.flush()

                start_time = time.time()
                # H = model.fit_spike_full_train(model, startEpoch, numberOfEpochs, ResNetModel, dataSet, sched, optimizer, loss_fn, train_loader, numberOfSteps, gClip, device, checkpointPeriod=1)
                H = model.fit_membrane_full_train(model, startEpoch, numberOfEpochs, ResNetModel, dataSet, sched, optimizer, loss_fn, train_loader, numberOfSteps, gClip, device, checkpointPeriod=1)
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"Training time: {execution_time:.4f} seconds")
            else:
                print("Train/valid split not defined")
                return None
            
            # final_membranes = {
            #     'mem1': model.mem1.detach().cpu(),
            #     'mem2': model.mem2.detach().cpu(),
            #     'mem3': model.mem3.detach().cpu(),
            #     'mem4': model.mem4.detach().cpu(),
            #     'mem5': model.mem5.detach().cpu(),
            #     # Add more if your model has more membrane states
            # }
            # torch.save(final_membranes, 'final_membranes.pth')

            weightPath = 'weights/spike/resnet' + str(ResNetModel) + '_weights_' + dataSet + '.pth'
            torch.save(model.state_dict(), weightPath)
            print("Training done.")
            return None
        
        else:
            print("Are we here")
            # For SEW ResNet
            # Define batch size
            batchSize = 32
            numberOfEpochs = 320
            workers = 16
            lr = 0.0025
            momentum = 0.9
            weightDecay = 0 

            # Use sync batch norm
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
            scaler = torch.amp.GradScaler()
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=320)

            print("Start training")

            for epoch in range(numberOfEpochs):
                
                # Train one epoch
                # for image, target in metric_logger.log_every(data_loader, print_freq, header):
                for batch_idx, (batch, labels) in enumerate(train_loader):
                    model.train()
                    # image, target = image.to(device), target.to(device)
                    batch = batch.to(device)
                    labels = labels.to(device)
                    # with torch.autograd.detect_anomaly():
                    if scaler is not None:
                        with torch.amp.autocast(device_type=device.type):
                            output = model(batch)
                            output = output.mean(dim=0)
                            # print(f"output shape: {output.shape}")
                            # print(f"labels shape: {labels.shape}")
                            loss = criterion(output, labels)
                    else:
                        output = model(batch)
                        loss = criterion(output, labels)
                    

                    optimizer.zero_grad()

                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    else:
                        loss.backward()
                        optimizer.step()

                    functional.reset_net(model)

                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    
                    # batch_size = batch.shape[0]
                    loss_s = loss.item()
                    if math.isnan(loss_s):
                        raise ValueError('loss is Nan')
                    acc1_s = acc1.item()
                    acc5_s = acc5.item()
                    print(f"Epoch: {epoch+1}/{numberOfEpochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc@1: {acc1.item():.2f}%, Acc@5: {acc5.item():.2f}%")

                lr_scheduler.step()

            weightPath = 'weights/spike/resnet' + str(ResNetModel) + '_weights_' + dataSet + '.pth'
            torch.save(model.state_dict(), weightPath)
            print("Training done.")
            return None


# TODO
def generate_latex(nameID, stats):
    if nameID == 'MNIST':
        namesOOD = ['FMNIST', 'KMNIST']
    elif nameID == 'FMNIST':
        namesOOD = ['MNIST', 'KMNIST']
    elif nameID == 'KMNIST':
        namesOOD = ['MNIST', 'FMNIST']
        
    latex_code = r"""
    \documentclass{article}
    \usepackage{amsmath}
    \begin{document}

    \title{Sample Document}
    \author{Author Name}
    \date{\today}
    \maketitle

    \section{Introduction}
    This is a sample LaTeX document generated using Python.

    \section{Math Example}
    Here is an example of a mathematical expression:
    \[
    E = mc^2
    \]

    \end{document}
    """

    # Save to a .tex file
    with open('sample_document.tex', 'w') as f:
        f.write(latex_code)
    