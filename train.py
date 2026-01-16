from utils.Utils import Utils
from torch.utils.data import random_split
import torch.nn as nn
import torch 
import matplotlib.pyplot as plt
from torchsummary import summary
from models.resnet import ResNet9Model , convNN4, ResNet10, ResNet18, newResNet10Model, newResNet18Model
from models.spikeresnet import spikeConvNN1, spikeConvNN2, spikeConvNN4,  SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model, SpikeResNet20Model  
from models.plain import spikeLinearNet1
import snntorch.functional as SF
import numpy as np
import sys
import time
import traceback
from snntorch import backprop
import gc
#from syops import get_model_complexity_info as syops_get_model_complexity_info
#from ptflops import get_model_complexity_info

# _seed_ = 1984
import random
import os

# Function to accumulate output sizes of Leaky layers
def accumulate_leaky_layer_outputs(model):
    total_output_size = 0
    for name, layer in model.named_modules():
        print(f"Checking layer: {name}, Type: {layer}")
        if "Leaky" in name:  # Check if the layer name contains "Leaky"
            if hasattr(layer, 'output_size'):
                output_size = layer.output_size  # Retrieve the output size
                layer_total = 1
                for dim in output_size:
                    layer_total *= dim  # Multiply dimensions to get total size
                total_output_size += layer_total
                print(f"Layer: {name}, Output Size: {output_size}, Accumulated: {layer_total}")
            else:
                print(f"Layer: {name} does not have an output_size attribute.")
    print(f"Total accumulated output size for Leaky layers: {total_output_size}")
    return total_output_size

def set_seed(_seed_):
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)  
    torch.cuda.manual_seed_all(_seed_) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 
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
    
    set_seed(config.seed)

    # Load dataset
    dataset_train, dataset_test = Utils.load_data(config.dataset_ID, config, auto_aug=config.auto_aug)

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, config.dataset_ID)
    # print(f"Image size: {channels, rows, cols}")
    #sys.stdout.flush() 
    
    if (config.full_train == False):
        train_loader, val_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, False, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config.seed))
    else:
        train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config.seed))

    # Get the device
    device = Utils.get_device()

    # Gradient clipping 
    gClip = config.gradient_clipping
    # Typical ranges
    # gClip = 0.5   # Common choice
    # gClip = 1.0   # Standard for many SNNs
    # gClip = 2.0   # For very deep networks

    # Weight decay
    wDecay = config.weight_decay

    # Learning rate
    lr = config.learning_rate

    # Get image size based on dataset
    if config.dataset_ID in ['CIFAR10', 'CIFAR100']:
        feature_size = 32
    elif config.dataset_ID in ['MNIST', 'FMNIST', 'KMNIST']:
        feature_size = 28
    else:
        feature_size = 28

    if config.model_type == 'conv':    

        # Define weight path similar to 'spike' case
        weightPath = 'weights/conv/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_' + str(config.num_time_steps_train) + '_E_' + str(config.expansion)+ '_L_'+str(config.loss)+ '_A_' + str(config.auto_aug) + '_S_' + str(config.seed) + '.pth'

        if (Utils.does_file_exists(weightPath)):
            # Keeps accuracy and loss for both training and validation in each epoch
            H = []

            # Define model
            if config.resnet_model == 4:
                model = convNN4(numberOfChannels=channels, 
                                numberOfClasses=config.num_classes, 
                                feature_size=32, 
                                expan=config.expansion)
            elif config.resnet_model == 10:
                model = ResNet10(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        expan=config.expansion)
            elif config.resnet_model == 18:
                model = ResNet18(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        expan=config.expansion  )
            else:
                print("Model not defined for the given ResNet configuration.")
                return -1
            
            #macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, backend='pytorch',
            #                               print_per_layer_stat=True, verbose=True)
            #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            #print('{:<30}  {:<8}'.format('Number of parameters: ', params))

            # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='aten'
            #                                         print_per_layer_stat=True, verbose=True)
            # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

            # Move model to device
            model = model.to(device)
            summary(model, input_size=(channels, rows, cols))

            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wDecay)

            # Scheduler
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=config.epochs, steps_per_epoch=len(train_loader))
            loss_fn = nn.CrossEntropyLoss()
            # Handle checkpointing
            startEpoch = 0
            weightsName = 'weights/conv/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_' + str(config.num_time_steps_train) + '_E_' + str(config.expansion)+ '_L_'+str(config.loss)+ '_A_' + str(config.auto_aug) + '_S_' + str(config.seed) + '_checkpoint.pth'
            if config.pretrained:
                print("Try to load checkpoint: " + weightsName)
                try:
                    file = torch.load(weightsName)
                    model.load_state_dict(file["model"])
                    optimizer.load_state_dict(file["optimizer"])
                    sched.load_state_dict(file["lr_scheduler"])
                    startEpoch = file["epochs"] + 1
                    print(f"numberOfEpochs: {config.epochs}, startEpoch: {startEpoch}")
                    print("Checkpoint loaded.")
                except Exception:
                    traceback.print_exc()
                    print("No valid checkpoint found. Starting from scratch.")

            # Training
            print("Training started")
            sys.stdout.flush()
            start_time = time.time()
            H = model.fit_conv_full_train(
                startEpoch=startEpoch,
                nEpochs=config.epochs,
                model=model,
                lossFunction=loss_fn,
                train_load=train_loader,
                ResNetModel=config.resnet_model,
                dataSet=config.dataset_ID,
                sched=sched,
                opt=optimizer,
                gd=gClip,
                device=device,
                checkpointFile=weightsName,
                checkpointPeriod=config.checkpointPeriod
            )
            end_time = time.time()  # Record end time
            execution_time = end_time - start_time  # Calculate execution time
            print(f"Training time: {execution_time:.4f} seconds")

            # Save final weights
            torch.save(model.state_dict(), weightPath)
            print("Training done.")

        else:
            print(f"Weights {weightPath} already exist.")

    elif config.model_type == 'spike':

        weightPath = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+ '_L_'+str(config.loss)+ '_A_'+str(config.auto_aug)+'_S_'+str(config.seed)+'.pth'
        # print(f"Weight path: {weightPath}")
        if (Utils.does_file_exists(weightPath)):

            # Keeps accuracy and loss for both training and validation in each epoch
            H = []

            beta = 0.95
            threshold = 0.25

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
                for name, param in model.named_parameters():
                    print(f"{name}: {param.data.norm().item()}")
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
                                        expansion=config.expansion)
                # model.reset_mem(batchSize, device)
            elif config.resnet_model == 18:
                model = SpikeResNet18Model(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        beta=beta, 
                                        threshold=threshold, 
                                        numberOfSteps=config.num_time_steps_train, 
                                        expansion=config.expansion)
            elif config.resnet_model == 20:
                model = SpikeResNet20Model(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        beta=beta, 
                                        threshold=threshold, 
                                        numberOfSteps=config.num_time_steps_train, 
                                        expansion=config.expansion)
            elif config.resnet_model == 21:
                model = spikeLinearNet1(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        beta=beta, 
                                        threshold=threshold)
            else:
                print("Model Not defined")
                return -1
            
            # radi samo na spikingjelly modelima 
            #ops, params = syops_get_model_complexity_info(model, (3, 32, 32), None, as_strings=True,
            #                                print_per_layer_stat=True, verbose=True)
            # print('{:<30}  {:<8}'.format('Computational complexity ACs:', acs))
            # print('{:<30}  {:<8}'.format('Computational complexity MACs:', macs))
            #print(f"ops: {ops}")
            #print('{:<30}  {:<8}'.format('Number of parameters: ', params))

            # nummm = accumulate_leaky_layer_outputs(model)
            # print(f"Total Leaky layer output size: {nummm}")
            
            # Move model to deviceS
            model = model.to(device)
            summary(model, input_size=(channels, rows, cols))

            # Loss function
            pop_code=False
            if config.expansion>1:
                pop_code=True
            if config.loss == 'rate_loss':
                # print(f"For population coding we do not use rate loss.")
                # return -1
                loss_fn = SF.ce_rate_loss(population_code=pop_code, num_classes=config.num_classes)
                loss_name = "ce_rate_loss"
            elif config.loss == 'count_loss':
                # print(f"For population coding we do not use count loss.")
                # return -1
                loss_fn = SF.ce_count_loss(population_code=pop_code, num_classes=config.num_classes)
                loss_name = "ce_count_loss"
            elif config.loss == 'cross_entropy':
                loss_fn = nn.CrossEntropyLoss()
                loss_name = "cross_entropy"
            elif config.loss == 'mse_count_loss':
                loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=pop_code, num_classes=config.num_classes)
                #loss_fn = SF.mse_count_loss()
                loss_name = "mse_count_loss"
            else:
                print("Loss function not defined")
                return -1
            # Optimizer for gray 5e-4
            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=wDecay)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wDecay)
            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=config.epochs, steps_per_epoch=len(train_loader))
            # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epochs)
            # sched = None

            # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epochs)
            # Training
            startEpoch = 0
            if (config.full_train == True):
                weightsName = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_L_'+str(config.loss)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed) + '_checkpoint_' + '.pth'
                if(config.pretrained == True):
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
                loss_name=""
                start_time = time.time()
                # Regular training fits according to the membrane voltages
                if config.fit == 'membrane':
                    H = model.fit_membrane_full_train(model, startEpoch, config.epochs, config.resnet_model, config.dataset_ID, sched, optimizer, loss_fn, loss_name, train_loader, config.num_time_steps_train, gClip, device, checkpointFile=weightsName,checkpointPeriod=config.checkpointPeriod)
                elif config.fit == 'spike':
                    H = model.fit_spike_full_train(model, startEpoch, config.epochs, config.resnet_model, config.dataset_ID, sched, optimizer, loss_fn, loss_name, train_loader, config.num_time_steps_train, gClip, device, checkpointFile=weightsName,checkpointPeriod=config.checkpointPeriod)
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

            torch.save(model.state_dict(), weightPath)
            print("Training done.")
            return None
        
        else:
            print("Weights " + str(weightPath) + " already exists.")
            return None



# def training_population(config):
#     '''
#     Training for the case: population coding + my spatio-temporal propagation
#     '''
    
#     def seed_worker(worker_id):
#         worker_seed = torch.initial_seed() % 2**32
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)

#     set_seed(config.seed)

#     # Load dataset
#     dataset_train, dataset_test = Utils.load_data(config.dataset_ID, config, auto_aug=config.auto_aug)

#     # Get image size
#     channels, rows, cols = Utils.get_image_size(dataset_train, config.dataset_ID)
#     print(f"Image size: {channels, rows, cols}")
#     sys.stdout.flush() 

#     print(f"Config batch size: {config.batch_size}")
    
#     if (config.full_train == False):
#         train_loader, val_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, False, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config.seed))
#     else:
#         train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config.seed))

#     # Get the device
#     device = Utils.get_device()

#     # Get image size based on dataset
#     if config.dataset_ID in ['CIFAR10', 'CIFAR100']:
#         feature_size = 32
#     elif config.dataset_ID in ['MNIST', 'FMNIST', 'KMNIST']:
#         feature_size = 28
#     else:
#         feature_size = 28  # default
#     print(f"Feature size: {feature_size}")

#     # Gradient clipping 
#     # gClip = 0.5
#     gClip = config.gradient_clipping

#     # Weight decay
#     wDecay = config.weight_decay

#     # Learning rate
#     lr = config.learning_rate

#     if config.model_type == 'conv':
#         # Define weight path for population coding
#         weightPath = 'weights/conv/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_' + str(config.num_time_steps_train) + '_E_' + str(config.expansion) + '_A_' + str(config.auto_aug) + '_S_' + str(config.seed) + '.pth'

#         if (Utils.does_file_exists(weightPath)):

#             print(f"Are we training...")

#             # Keeps accuracy and loss for both training and validation in each epoch
#             H = []

#             # Define model
#             if config.resnet_model == 4:
#                 model = convNN4(numberOfChannels=channels, 
#                                 numberOfClasses=config.num_classes, 
#                                 feature_size=32, 
#                                 expansion=config.expansion)
#             elif config.resnet_model == 10:
#                 model = ResNet10(numberOfChannels=channels, 
#                                         numberOfClasses=config.num_classes, 
#                                         expan=config.expansion)
#             elif config.resnet_model == 18:
#                 model = ResNet18(numberOfChannels=channels, 
#                                         numberOfClasses=config.num_classes, 
#                                         expan=config.expansion)
#             else:
#                 print("Model not defined for the given ResNet configuration.")
#                 return -1
            
#             print(model)

#             # Move model to device
#             model = model.to(device)
#             print(f"In this moment the batch size is: {config.batch_size}, channels: {channels}, rows: {rows}, cols: {cols}")
#             # summary(model, batch_size=config.batch_size, input_size=(channels, rows, cols))

#             # Loss function
#             loss_fn = nn.CrossEntropyLoss()
            
#             # Optimizer
#             optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wDecay)

#             # Scheduler
#             sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=config.epochs, steps_per_epoch=len(train_loader))

#             # Handle checkpointing
#             startEpoch = 0
#             if config.pretrained:
#                 weightsName = 'weights/conv/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_' + str(config.num_time_steps_train) + '_E_' + str(config.expansion) + '_A_' + str(config.auto_aug) + '_S_' + str(config.seed) + '_pop_checkpoint.pth'
#                 print("Try to load checkpoint: " + weightsName)
#                 try:
#                     file = torch.load(weightsName)
#                     model.load_state_dict(file["model"])
#                     optimizer.load_state_dict(file["optimizer"])
#                     sched.load_state_dict(file["lr_scheduler"])
#                     startEpoch = file["epochs"] + 1
#                     print(f"numberOfEpochs: {config.epochs}, startEpoch: {startEpoch}")
#                     print("Checkpoint loaded.")
#                 except Exception:
#                     traceback.print_exc()
#                     print("No valid checkpoint found. Starting from scratch.")

#             # Training
#             print("Training started")
#             sys.stdout.flush()
#             start_time = time.time()
#             H = model.fit_conv_full_train(
#                 startEpoch=startEpoch,
#                 nEpochs=config.epochs,
#                 model=model,
#                 lossFunction=loss_fn,
#                 train_load=train_loader,
#                 ResNetModel=config.resnet_model,
#                 dataSet=config.dataset_ID,
#                 sched=sched,
#                 opt=optimizer,
#                 gd=gClip,
#                 device=device,
#                 checkpointPeriod=1
#             )
#             end_time = time.time()  # Record end time
#             execution_time = end_time - start_time  # Calculate execution time
#             print(f"Training time: {execution_time:.4f} seconds")

#             # Save final weights
#             torch.save(model.state_dict(), weightPath)
#             print("Training done.")

#         else:
#             print(f"Weights {weightPath} already exist.")

#     elif config.model_type == 'spike':

#         # Check if the network is already trained:
#         weightPath = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_L_'+str(config.loss)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed)+'.pth'
#         if (Utils.does_file_exists(weightPath)):

#             # Keeps accuracy and loss for both training and validation in each epoch
#             H = []

#             beta = 0.95
#             threshold = 0.25

#             # Define model
#             if config.resnet_model == 2:
#                 model = spikeConvNN2(numberOfChannels=channels, 
#                                     numberOfClasses=config.num_classes, 
#                                     beta=beta, 
#                                     threshold=threshold, 
#                                     feature_size=feature_size,
#                                     numberOfSteps=config.num_time_steps_train, 
#                                     expansion=config.expansion)
#             elif config.resnet_model == 4:
#                 model = spikeConvNN4(numberOfChannels=channels, 
#                                     numberOfClasses=config.num_classes, 
#                                     beta=beta, 
#                                     threshold=threshold, 
#                                     feature_size=feature_size, 
#                                     numberOfSteps=config.num_time_steps_train, 
#                                     expansion=config.expansion)
#             elif config.resnet_model == 10:
#                 model = SpikeResNet10Model(numberOfChannels=channels, 
#                                             numberOfClasses=config.num_classes, 
#                                             beta=beta, 
#                                             threshold=threshold, 
#                                             numberOfSteps=config.num_time_steps_train, 
#                                             expansion=config.expansion)
#             elif config.resnet_model == 18:
#                 model = SpikeResNet18Model(numberOfChannels=channels, 
#                                             numberOfClasses=config.num_classes, 
#                                             beta=beta, 
#                                             threshold=threshold, 
#                                             numberOfSteps=config.num_time_steps_train, 
#                                             expansion=config.expansion)
#             else:
#                 print("Train Not defined")
#                 return -1
            
#             model = model.to(device)
#             summary(model, input_size=(channels, rows, cols))
#             loss_name=""
#             pop_code=False
#             if config.expansion>1:
#                 pop_code=True
#             if config.loss == 'rate_loss':
#                 # print(f"For population coding we do not use rate loss.")
#                 # return -1
#                 loss_fn = SF.ce_rate_loss(population_code=pop_code, num_classes=config.num_classes)
#                 loss_name = "ce_rate_loss"
#             elif config.loss == 'count_loss':
#                 # print(f"For population coding we do not use count loss.")
#                 # return -1
#                 loss_fn = SF.ce_count_loss(population_code=pop_code, num_classes=config.num_classes)
#                 loss_name = "ce_count_loss"
#             elif config.loss == 'cross_entropy':
#                 loss_fn = nn.CrossEntropyLoss()
#                 loss_name = "cross_entropy"
#             elif config.loss == 'mse_count_loss':
#                 loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=pop_code, num_classes=config.num_classes)
#                 #loss_fn = SF.mse_count_loss()
#                 loss_name = "mse_count_loss"
            
#             optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
#             sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=config.epochs, steps_per_epoch=len(train_loader))

#             # max_lr = 1e-2
#             # div_factor = 5.0
#             # initial_lr = max_lr / div_factor  # = 2e-3

#             # optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999)) 
#             # sched = torch.optim.lr_scheduler.OneCycleLR(
#             #     optimizer, 
#             #     max_lr=max_lr,           
#             #     epochs=epochs, 
#             #     steps_per_epoch=len(train_loader),
#             #     pct_start=0.3,
#             #     div_factor=div_factor,        
#             #     final_div_factor=100   
#             # )

#             # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
#             # sched = torch.optim.lr_scheduler.CosineAnnealingLR(
#             #     optimizer, 
#             #     T_max=epochs,
#             #     eta_min=1e-5  # Minimum LR
#             # )

#             # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
#             # sched = torch.optim.lr_scheduler.StepLR(
#             #     optimizer, 
#             #     step_size=1,     # Reduce LR every 10 epochs
#             #     gamma=0.5         # Multiply by 0.5
#             # )

#             # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
#             # sched = torch.optim.lr_scheduler.ExponentialLR(
#             #     optimizer, 
#             #     gamma=0.95        # Multiply by 0.95 each epoch
#             # )

#             # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
#             # sched = None 

#             # Handle checkpointing
#             if (config.full_train == True):
#                 print(f"Pretrained: {config.pretrained}")
#                 if(config.pretrained == True):
#                     weightsName = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_L_'+str(config.loss)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed) + '_checkpoint_' + '.pth'
#                     print("Try to load checkpoint: " + weightsName)
#                     try:
#                         file = torch.load(weightsName)
#                         model.load_state_dict(file["model"])
#                         optimizer.load_state_dict(file["optimizer"])
#                         if sched is not None:
#                             sched.load_state_dict(file["lr_scheduler"])
#                         startEpoch = file["epochs"] + 1
#                         print(f"numberOfEpochs: {config.epochs}, startEpoch: {startEpoch}")
#                         print("Checkpoint loaded.")
#                     except Exception:
#                         traceback.print_exc()
#                         print("No valid checkpoint found. Starting from scratch.")
#                 startEpoch = 0
#                 print("Training started")
#                 sys.stdout.flush()
                
#                 start_time = time.time()
#                 if config.fit == 'membrane':
#                     # print(f"For population coding we can not fit on membrane voltage.")
#                     # return -1 
#                     H = model.fit_membrane_full_train(model, startEpoch, config.epochs, config.resnet_model, config.dataset_ID, sched, optimizer, loss_fn, loss_name, train_loader, config.num_time_steps_train, gClip, device, checkpointPeriod=1)
#                 elif config.fit == 'spike':
#                     H = model.fit_spike_full_train(model, startEpoch, config.epochs, config.resnet_model, config.dataset_ID, sched, optimizer, loss_fn, loss_name, train_loader, config.num_time_steps_train, gClip, device, checkpointPeriod=1)
#                 end_time = time.time()  # Record end time
#                 execution_time = end_time - start_time  # Calculate execution time
#                 print(f"Training time: {execution_time:.4f} seconds")
#             else:
#                 print("Train/valid split not defined")
#                 return None

#             torch.save(model.state_dict(), weightPath)
#             del model 
#             del train_loader, test_loader
#             gc.collect()
#             torch.cuda.empty_cache()
#             print("Training done.")

#         return None
#     else:
#         print("Weights " + str(weightPath) + " already exists.")
#         return None
