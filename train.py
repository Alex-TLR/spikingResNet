from utils.Utils import Utils
#from metrics.Metrics import Metrics
from torch.utils.data import random_split
import torch.nn as nn
import torch 
import matplotlib.pyplot as plt
from torchsummary import summary
from models.resnet import ResNet9Model 
from models.spikeresnet import spikeConvNN1, spikeConvNN2, spikeConvNN4,  SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model, SpikeResNet20Model  
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
from snntorch import backprop
from test import test_accuracy_population

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


def training(config):
    '''
    dataSet:        defines the data set for training (for example MNIST, FMNIST, KMNIST)
    modelType:      convolutional or spiking neural network
    expansion:      defines the expansion factor for ResNet models when population coding is used
                    in this training case ut is set to 1
    case:           case needs to contain the details of the case scenario
    fullTrain:      define if training is done on complete training set or train/valid split is used
    ResNetModel:    determines number of layers in model

    This training approach use the membrane voltage for loss function
    No population coding is used, so expansion is set to 1
    Number of training steps is set to 4 to compare with the results from the literature
    '''

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Load dataset
    dataset_train, dataset_test = Utils.load_data(config.dataset_ID, auto_aug=config.auto_aug)

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, config.dataset_ID)
    # print(f"Image size: {channels, rows, cols}")
    sys.stdout.flush() 
    
    if (config.full_train == False):
        train_loader, val_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, False, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(_seed_))
    else:
        train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(_seed_))

    # Get the device
    device = Utils.get_device()

    # Gradient clipping 
    gClip = 0.5
    # Typical ranges
    # gClip = 0.5   # Common choice
    # gClip = 1.0   # Standard for many SNNs
    # gClip = 2.0   # For very deep networks

    # Weight decay
    wDecay = 0.0001

    if config.model_type == 'conv':    

        # Learning rate (KEY)
        lr = [0.0001]

        # Number of epochs
        numberOfEpochs = [20]

        # Loss function
        lossFunction = nn.CrossEntropyLoss()

        # Model
        if config.resnet_model == 9:
            model = ResNet9Model(numberOfChannels=channels, numberOfClasses=config.num_classes)
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
            if (config.full_train == False):
                model, H = model.fit_conv(nEpochs, model, lossFunction, lrCurrent, train_loader, val_loader, H, device, wd=wDecay, gd=gClip)
            elif (config.full_train == True):
                model, H = model.fit_conv_full_train(nEpochs, model, lossFunction, lrCurrent, train_loader, H, device, wd=wDecay, gd=gClip)
        print("\n")

        weightPath = 'weights/conv/resnet9_weights_' + config.dataset_ID + '.pth'
        torch.save(model.state_dict(), weightPath)

        if (config.full_train == False):
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

    elif config.model_type == 'spike':

        weightPath = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_A_'+str(config.auto_aug)+'.pth'
        if (Utils.does_file_exists(weightPath)):

            # Keeps accuracy and loss for both training and validation in each epoch
            H = []

            beta = 0.95
            threshold = 0.25

            # Get image size based on dataset
            if config.dataset_ID in ['CIFAR10', 'CIFAR100']:
                feature_size = 32
            elif config.dataset_ID in ['MNIST', 'FMNIST', 'KMNIST']:
                feature_size = 28
            else:
                feature_size = 28

            # Define model
            if config.resnet_model == 1:
                model = spikeConvNN1(numberOfChannels=channels, 
                                    numberOfClasses=config.num_classes, 
                                    beta=beta, 
                                    threshold=threshold)
            elif config.resnet_model == 2:
                model = spikeConvNN2(numberOfChannels=channels, 
                                    numberOfClasses=config.num_classes, 
                                    beta=beta, 
                                    threshold=threshold, 
                                    feature_size=feature_size)
                # model.reset_mem(batchSize, device)
            elif config.resnet_model == 4:
                model = spikeConvNN4(numberOfChannels=channels, 
                                    numberOfClasses=config.num_classes, 
                                    beta=beta, 
                                    threshold=threshold, 
                                    feature_size=feature_size, 
                                    numberOfSteps=config.num_time_steps_train, 
                                    expansion=config.expansion)
            elif config.resnet_model == 9:
                model = SpikeResNet9Model(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        beta=beta, 
                                        threshold=threshold)
            elif config.resnet_model == 10:
                model = SpikeResNet10Model(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        beta=beta, 
                                        threshold=threshold, 
                                        numberOfSteps=config.num_time_steps_train, 
                                        expansion=1)
                # model.reset_mem(batchSize, device)
            elif config.resnet_model == 18:
                model = SpikeResNet18Model(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        beta=beta, 
                                        threshold=threshold, 
                                        numberOfSteps=config.num_time_steps_train, 
                                        expansion=1)
            elif config.resnet_model == 20:
                model = SpikeResNet20Model(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        beta=beta, 
                                        threshold=threshold, 
                                        numberOfSteps=config.num_time_steps_train, 
                                        expansion=1)
            elif config.resnet_model == 21:
                model = spikeLinearNet1(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        beta=beta, 
                                        threshold=threshold)
            else:
                print("Not defined")
                return -1
            
            # Move model to device
            model = model.to(device)

            # Loss function
            if config.loss == 'rate_loss':
                loss_fn = SF.ce_rate_loss()
            elif config.loss == 'count_loss':
                loss_fn = SF.ce_count_loss()
            elif config.loss == 'cross_entropy':
                loss_fn = nn.CrossEntropyLoss()
            # Optimizer for gray 5e-4
            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=wDecay)
            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=wDecay)
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=config.epochs, steps_per_epoch=len(train_loader))
            # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epochs)
            # sched = None

            # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epochs)
            # Training
            startEpoch = 0
            if (config.full_train == True):
                if(config.pretrained == True):
                    weightsName = 'weights/spike/' + 'resnet_' + str(config.resnet_model) + '_' + config.dataset_ID + '_T_' + str(config.num_time_steps_train) + '_E_' + str(config.expansion) + '_checkpoint_' + '.pth'
                    print("Try to load checkpoint: "+weightsName)
                    try:
                        file = torch.load(weightsName)
                        model.load_state_dict(file["model"])
                        optimizer.load_state_dict(file["optimizer"])
                        sched.load_state_dict(file["lr_scheduler"])
                        startEpoch=file["epochs"] + 1
                        print(f"numberOfEpochs: {config.epochs}, startEpoch: {startEpoch}, steps_per_epoch: {len(train_loader)}, total_steps: {sched.total_steps}")
                        #sched._step_count = startEpoch * len(train_loader)
                        print("Checkpoint loaded.")
                    except Exception:
                        traceback.print_exc()
                        print("No valid checkpoint found. Starting from scratch.")
                    print("Training started")
                    sys.stdout.flush()

                start_time = time.time()
                # Regular training fits according to the membrane voltages
                if config.fit == 'membrane':
                    H = model.fit_membrane_full_train(model, startEpoch, config.epochs, config.resnet_model, config.dataset_ID, sched, optimizer, loss_fn, train_loader, config.num_time_steps_train, gClip, device, checkpointPeriod=1)
                elif config.fit == 'spike': 
                    H = model.fit_spike_full_train(model, startEpoch, config.epochs, config.resnet_model, config.dataset_ID, sched, optimizer, loss_fn, train_loader, config.num_time_steps_train, gClip, device, checkpointPeriod=1)
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

            # weightPath = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_A_'+str(config.auto_aug)+'.pth'
            torch.save(model.state_dict(), weightPath)
            print("Training done.")
            return None
        
        else:
            print("Weights " + str(weightPath) + " already exists.")
            return None



# TODO: needs to be removed or revised
def training_population(dataSet, modelType, batchSize, numOfClasses, ResNetModel, epochs=200, fullTrain=False, auto_aug=False, expansion=1, pretrained=False):
    '''
    Training for the case: population coding + BPTT
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

    # Get image size based on dataset
    if dataSet in ['CIFAR10', 'CIFAR100']:
        feature_size = 32
    elif dataSet in ['MNIST', 'FMNIST', 'KMNIST']:
        feature_size = 28
    else:
        feature_size = 28  # default
    print(f"Feature size: {feature_size}")

    # Gradient clipping 
    gClip = 0.5

    # Weight decay
    wDecay = 0.0001

    if modelType == 'conv':
        pass 

    elif modelType == 'spike':

        # For spiking neural network we need number of steps
        numberOfSteps = 1
        beta = 0.95
        threshold = 0.25

        # Define model
        if ResNetModel == 4:
            model = spikeConvNN4(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold, feature_size=feature_size)
            model = model.to(device)
        else:
            print("Not defined")
            return -1

        loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=numberOfClasses)

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=epochs, steps_per_epoch=len(train_loader))

        # max_lr = 1e-2
        # div_factor = 5.0
        # initial_lr = max_lr / div_factor  # = 2e-3

        # optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999))
        # sched = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, 
        #     max_lr=max_lr,           
        #     epochs=epochs, 
        #     steps_per_epoch=len(train_loader),
        #     pct_start=0.3,
        #     div_factor=div_factor,        
        #     final_div_factor=100   
        # )

        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wDecay)

        # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999), weight_decay=wDecay)
        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 
        #     T_max=epochs,
        #     eta_min=1e-5  # Minimum LR
        # )

        # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
        # sched = torch.optim.lr_scheduler.StepLR(
        #     optimizer, 
        #     step_size=10,     # Reduce LR every 10 epochs
        #     gamma=0.5         # Multiply by 0.5
        # )

        # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
        # sched = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, 
        #     gamma=0.95        # Multiply by 0.95 each epoch
        # )

        # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
        # sched = None 

        # Handle checkpointing
        if (fullTrain == True):
            print(f"Pretrained: {pretrained}")
            if(pretrained == True):
                weightsName = 'weights/spike/' + 'resnet' + str(ResNetModel) + dataSet + '_checkpoint_' + '.pth'
                print("Try to load checkpoint: " + weightsName)
                try:
                    file = torch.load(weightsName)
                    model.load_state_dict(file["model"])
                    optimizer.load_state_dict(file["optimizer"])
                    if sched is not None:
                        sched.load_state_dict(file["lr_scheduler"])
                    startEpoch = file["epochs"] + 1
                    print(f"numberOfEpochs: {numberOfEpochs}, startEpoch: {startEpoch}")
                    print("Checkpoint loaded.")
                except Exception:
                    traceback.print_exc()
                    print("No valid checkpoint found. Starting from scratch.")
            startEpoch = 0
            print("Training started")
            sys.stdout.flush()
            
            # Start timing
            start_time = time.time()
            
            # Use snnTorch BPTT for simplified training
            print(f"Starting training from epoch {startEpoch} to {numberOfEpochs}")
            
            for epoch in range(startEpoch, numberOfEpochs):
                # Reset network state before each epoch
                # utils.reset(model)

                # Train one epoch using BPTT
                avg_loss = backprop.BPTT(
                    net=model,
                    dataloader=train_loader, 
                    num_steps=numberOfSteps,
                    optimizer=optimizer, 
                    criterion=loss_fn, 
                    time_var=False,  # Same input at each time step
                    device=device
                )

                # print(f"After BPTT - LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Update learning rate scheduler if used
                if sched is not None:
                    sched.step()
                    # print(f"After scheduler step - LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Print progress
                # current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch+1}/{numberOfEpochs}, Loss: {avg_loss:.4f}")
                test_accuracy_population(dataSet, model, modelType, batchSize, numberOfClasses, ResNetModel)

                # Save checkpoint periodically (every 10 epochs)
                if (epoch + 1) % 10 == 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': sched.state_dict() if sched is not None else None,
                        'epochs': epoch,
                        'loss': avg_loss
                    }
                    checkpoint_path = f'weights/spike/resnet{ResNetModel}{dataSet}_checkpoint_.pth'
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Checkpoint saved at epoch {epoch+1}")
                
                sys.stdout.flush()
            
            # Calculate and print training time
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Training time: {execution_time:.4f} seconds")
        else:
            print("Train/valid split not defined")
            return None
        
        weightPath = 'weights/spike/resnet' + str(ResNetModel) + '_weights_' + dataSet +'_T_'+str(numberOfSteps)+'_E_'+str(expansion)+'_A_'+str(auto_aug)+'.pth'

        torch.save(model.state_dict(), weightPath)
        print("Training done.")
        print(f"Final weights saved to: {weightPath}")
    return None


def training_population_2(config):
    '''
    Training for the case: population coding + my spatio-temporal propagation
    '''
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Load dataset
    dataset_train, dataset_test = Utils.load_data(config.dataset_ID, auto_aug=config.auto_aug)

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, config.dataset_ID)
    # print(f"Image size: {channels, rows, cols}")
    sys.stdout.flush() 
    
    if (config.full_train == False):
        train_loader, val_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, False, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(_seed_))
    else:
        train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(_seed_))

    # Get the device
    device = Utils.get_device()

    # Get image size based on dataset
    if config.dataset_ID in ['CIFAR10', 'CIFAR100']:
        feature_size = 32
    elif config.dataset_ID in ['MNIST', 'FMNIST', 'KMNIST']:
        feature_size = 28
    else:
        feature_size = 28  # default
    print(f"Feature size: {feature_size}")

    # Gradient clipping 
    gClip = 0.5

    # # Weight decay
    # wDecay = 0.0001

    if config.model_type == 'conv':
        pass 

    elif config.model_type == 'spike':

        # Check if the network is already trained:
        weightPath = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_A_'+str(config.auto_aug)+'.pth'
        if (Utils.does_file_exists(weightPath)):

            # Keeps accuracy and loss for both training and validation in each epoch
            H = []

            beta = 0.95
            threshold = 0.25

            # Define model
            if config.resnet_model == 2:
                model = spikeConvNN2(numberOfChannels=channels, 
                                    numberOfClasses=config.num_classes, 
                                    beta=beta, 
                                    threshold=threshold, 
                                    feature_size=feature_size,
                                    numberOfSteps=config.num_time_steps_train, 
                                    expansion=config.expansion)
            elif config.resnet_model == 4:
                model = spikeConvNN4(numberOfChannels=channels, 
                                    numberOfClasses=config.num_classes, 
                                    beta=beta, 
                                    threshold=threshold, 
                                    feature_size=feature_size, 
                                    numberOfSteps=config.num_time_steps_train, 
                                    expansion=config.expansion)
            elif config.resnet_model == 10:
                model = SpikeResNet10Model(numberOfChannels=channels, 
                                            numberOfClasses=config.num_classes, 
                                            beta=beta, 
                                            threshold=threshold, 
                                            numberOfSteps=config.num_time_steps_train, 
                                            expansion=config.expansion)
            elif config.resnet_model == 18:
                model = SpikeResNet18Model(numberOfChannels=channels, 
                                            numberOfClasses=config.num_classes, 
                                            beta=beta, 
                                            threshold=threshold, 
                                            numberOfSteps=config.num_time_steps_train, 
                                            expansion=config.expansion)
            else:
                print("Train Not defined")
                return -1
            
            model = model.to(device)

            if config.loss == 'rate_loss':
                print(f"For population coding we do not use rate loss.")
                return -1
                # loss_fn = SF.ce_rate_loss()
            elif config.loss == 'count_loss':
                print(f"For population coding we do not use count loss.")
                return -1
                # loss_fn = SF.ce_count_loss()
            elif config.loss == 'cross_entropy':
                print(f"For population coding we do not use cross entropy loss.")
                return -1
                # loss_fn = nn.CrossEntropyLoss()
            elif config.loss == 'mse_count_loss':
                loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=config.num_classes)

            optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=config.epochs, steps_per_epoch=len(train_loader))


            # max_lr = 1e-2
            # div_factor = 5.0
            # initial_lr = max_lr / div_factor  # = 2e-3

            # optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999)) 
            # sched = torch.optim.lr_scheduler.OneCycleLR(
            #     optimizer, 
            #     max_lr=max_lr,           
            #     epochs=epochs, 
            #     steps_per_epoch=len(train_loader),
            #     pct_start=0.3,
            #     div_factor=div_factor,        
            #     final_div_factor=100   
            # )

            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
            # sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, 
            #     T_max=epochs,
            #     eta_min=1e-5  # Minimum LR
            # )

            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
            # sched = torch.optim.lr_scheduler.StepLR(
            #     optimizer, 
            #     step_size=1,     # Reduce LR every 10 epochs
            #     gamma=0.5         # Multiply by 0.5
            # )

            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
            # sched = torch.optim.lr_scheduler.ExponentialLR(
            #     optimizer, 
            #     gamma=0.95        # Multiply by 0.95 each epoch
            # )

            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
            # sched = None 

            # Handle checkpointing
            if (config.full_train == True):
                print(f"Pretrained: {config.pretrained}")
                if(config.pretrained == True):
                    weightsName = 'weights/spike/' + 'resnet_' + str(config.resnet_model) + '_' + config.dataset_ID + '_T_' + str(config.num_time_steps_train) + '_E_' + str(config.expansion) + '_checkpoint_' + '.pth'
                    print("Try to load checkpoint: " + weightsName)
                    try:
                        file = torch.load(weightsName)
                        model.load_state_dict(file["model"])
                        optimizer.load_state_dict(file["optimizer"])
                        if sched is not None:
                            sched.load_state_dict(file["lr_scheduler"])
                        startEpoch = file["epochs"] + 1
                        print(f"numberOfEpochs: {config.epochs}, startEpoch: {startEpoch}")
                        print("Checkpoint loaded.")
                    except Exception:
                        traceback.print_exc()
                        print("No valid checkpoint found. Starting from scratch.")
                startEpoch = 0
                print("Training started")
                sys.stdout.flush()
                
                start_time = time.time()
                if config.fit == 'membrane':
                    print(f"For population coding we can not fit on membrane voltage.")
                    return -1 
                    # H = model.fit_membrane_full_train(model, startEpoch, config.epochs, config.resnet_model, config.dataset_ID, sched, optimizer, loss_fn, train_loader, config.num_time_steps_train, gClip, device, checkpointPeriod=1)
                elif config.fit == 'spike':
                    H = model.fit_spike_full_train(model, startEpoch, config.epochs, config.resnet_model, config.dataset_ID, sched, optimizer, loss_fn, train_loader, config.num_time_steps_train, gClip, device, checkpointPeriod=1)
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"Training time: {execution_time:.4f} seconds")
            else:
                print("Train/valid split not defined")
                return None

            # weightPath = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_A_'+str(config.auto_aug)+'.pth'

            torch.save(model.state_dict(), weightPath)
            print("Training done.")

        return None
    else:
        print("Weights " + str(weightPath) + " already exists.")
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
    