from logging import config
from xml.parsers.expat import model
import numpy as np
import time 
import os
from utils.Utils import Utils
from metrics.Metrics import Metrics
from clustering.Clustering import Clustering
from models.spikeresnet import spikeConvNN1, spikeConvNN2, spikeConvNN4, SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model, SpikeResNet20Model  
from models.plain import spikeLinearNet1
from models.resnet import convNN4, ResNet10, ResNet18, newResNet10Model, newResNet18Model
import torch 
import snntorch.functional as SF 
import matplotlib.pyplot as plt
from snntorch import utils
import gc


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


def test_accuracy(config):
    '''
    check accuracy of trained model on ID test data
    '''

    # Load database
    # print(f"dataSet: {dataSet}")
    dataset_train, dataset_test = Utils.load_data(config.dataset_ID, config)
    testSize = len(dataset_test)
    # print(f"Number of test set images is: {testSize}")

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, config.dataset_ID)

    # Load data
    train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, True)

    # Get the device
    device = Utils.get_device()
    # device = torch.device("cpu")

    # Get image size based on dataset
    if config.dataset_ID in ['CIFAR10', 'CIFAR100']:
        feature_size = 32
    elif config.dataset_ID in ['MNIST', 'FMNIST', 'KMNIST']:
        feature_size = 28
    else:
        feature_size = 28  # default

    if config.model_type == 'conv':
        # Define model
        if config.resnet_model == 4:
            model = convNN4(numberOfChannels=channels, 
                            numberOfClasses=config.num_classes, 
                            feature_size=32, 
                            expansion=config.expansion)
        elif config.resnet_model == 10:
            model = ResNet10(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        expansion=config.expansion)
        elif config.resnet_model == 18:
            model = ResNet18(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        expansion=config.expansion)
        else:
            print("Model not defined for the given ResNet configuration.")
            return -1

        # Load weights
        weightsName = 'weights/conv/exp'+str(config.case)+'/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_' + str(config.num_time_steps_train) + '_E_' + str(config.expansion) + '_L_'+str(config.loss)+ '_A_' + str(config.auto_aug) + '_S_' + str(config.seed) + '.pth'
        if not os.path.exists(weightsName):
            print(f"Weights file not found: {weightsName}")
            return -1

        model.load_state_dict(torch.load(weightsName, weights_only=False))
        model = model.to(device)

        # Evaluate model
        model.eval()
        total_correct_membrane = 0
        total_samples = 0
        with torch.no_grad():
            for batch, labels in test_loader:
                batch = batch.to(device)
                labels = labels.to(device)

                # Forward pass
                mem, _ = model(batch)
                # print(f"Membrane shape: {membrane.shape}")
                if model.expansion > 1:
                    mem = mem.reshape(mem.shape[0], model.numberOfClasses, model.expansion).sum(dim=2)
                predicted = torch.argmax(mem, dim=1)
                correct = (predicted == labels).float()
                total_correct_membrane += correct.sum()
                total_samples += batch.size(0)

        acc_membrane = total_correct_membrane / total_samples * 100
        # print(f"Inside test.py: {acc_membrane}")
        return None, acc_membrane

    elif config.model_type == 'spike':

        # For spiking neural network we need number of steps
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
            model = spikeLinearNet1(numberOfChannels=channels, numberOfClasses=config.num_classes, beta=beta, threshold=threshold)
        else:
            print("Model Not defined")
            return -1
        
        # Load weights
        model = model.to(device)
        torch.cuda.empty_cache()

        weightsName = 'weights/spike/exp'+str(config.case)+'/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_L_'+str(config.loss)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed)+'.pth'
        if not os.path.exists(weightsName):
            # try alternate filename without the loss token (legacy name)
            wN = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed)+'.pth'
            if os.path.exists(wN):
                print(f"Found legacy weights file: {wN}, renaming to new format.")
                try:
                    os.rename(wN, weightsName)
                    print(f"Renamed legacy weights file {wN} -> {weightsName}")
                except Exception as e:
                    print(f"Warning: failed to rename {wN} -> {weightsName}: {e}")
            else:
                print(f"Weights file not found: {weightsName}")
                return -1, -1
        # print(f"weightsName: {weightsName}")
        model.load_state_dict(torch.load(weightsName, weights_only=False))
        model = model.to(device)

        # print("Check test images.")
        testLen = 0
        i = 0
        # print(f"Number of batches: {len(test_loader)}")
        model.eval()
        
        total_correct_spikes = 0
        total_correct_membrane = 0
        total_samples = 0
        population_code = False
        if config.expansion > 1:
            population_code = True
        with torch.no_grad():
            for batch, labels in test_loader:
                testLen += len(batch)
                batch = batch.to(device)
                la = labels[0].item()
                labels = labels.to(device)
                # model.mem1 = final_membranes['mem1'].to(device)
                # model.mem2 = final_membranes['mem2'].to(device)
                # model.mem3 = final_membranes['mem3'].to(device)
                # model.mem4 = final_membranes['mem4'].to(device)
                # model.mem5 = final_membranes['mem5'].to(device)
                # Generate predictions/ forward pass
                spikes, _, membrane, _ = model(batch, config.num_time_steps_extract)
                # print(f"spikes.max: {spikes.max()}")
                # l_spikes = loss_fn(spikes, labels)
                # testLossSpikes.append(l_spikes.item())
                # a1, a2 = model.accuracy_spike(model, numberOfSteps, batch, labels, device)
                # print(f"testLen: {testLen}, a1: {a1}, {a1.item()}, a2: {a2}, {a2.item()}")
                # testAcc[0, i] = a2.item()
                # testAccList.append(a1.item())
                # On spikes
                batch_size = batch.size(0)
                acc = SF.accuracy_rate(spikes, labels, population_code=population_code, num_classes=model.numberOfClasses) 
                acc *= batch_size
                total_correct_spikes += acc

                # On membrane
                mem = membrane.mean(0)
                acc_rate_mem = SF.accuracy_rate(
                    mem.unsqueeze(0),  # Add time dimension [1, B, classes*expansion]
                    labels, 
                    population_code=population_code, 
                    num_classes=model.numberOfClasses
                )
                batch_correct_mem = (acc_rate_mem * mem.size(0)).item()
                total_correct_membrane += batch_correct_mem

                total_samples += batch_size
                
                print(f"\rCurrent: {total_correct_spikes}, Test accuracy on spikes: {total_correct_spikes/total_samples*100:05.2f}, Test accuracy on membrane: {total_correct_membrane/total_samples*100:05.2f}, progress: {i+1}/{len(test_loader)}", end='', flush=True)
                i += 1

        print("\nDone.")
        acc_membrane = total_correct_membrane / total_samples * 100
        acc_spikes = total_correct_spikes / total_samples * 100

        del model 
        del train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
        return acc_spikes, acc_membrane


# def test_accuracy_population(config):
#     '''
#     check accuracy of trained model on ID test data when population coding is used with my spatio-temporal propagation
#     '''
#     # Load database
#     dataset_train, dataset_test = Utils.load_data(config.dataset_ID, config)
#     testSize = len(dataset_test)

#     # Get image size
#     channels, rows, cols = Utils.get_image_size(dataset_train, config.dataset_ID)

#     # Load data
#     train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, True)

#     # Get the device
#     device = Utils.get_device()

#     # Get image size based on dataset
#     if config.dataset_ID in ['CIFAR10', 'CIFAR100']:
#         feature_size = 32
#     elif config.dataset_ID in ['MNIST', 'FMNIST', 'KMNIST']:
#         feature_size = 28
#     else:
#         feature_size = 28

#     if config.model_type == 'spike':
#         # For spiking neural network we need number of steps
#         numberOfSteps = config.num_time_steps_train
#         beta = 0.95
#         threshold = 0.25

#         # Define model with expansion parameter
#         if config.resnet_model == 2:
#             model = spikeConvNN2(
#                 numberOfChannels=channels, 
#                 numberOfClasses=config.num_classes, 
#                 beta=beta, 
#                 threshold=threshold, 
#                 expansion=config.expansion, 
#                 feature_size=feature_size
#             )
#         elif config.resnet_model == 4:
#             model = spikeConvNN4(numberOfChannels=channels, 
#                                  numberOfClasses=config.num_classes, 
#                                  beta=beta, threshold=threshold, 
#                                  feature_size=feature_size, 
#                                  numberOfSteps=numberOfSteps, 
#                                  expansion=config.expansion)     
#         elif config.resnet_model == 10:
#             model = SpikeResNet10Model(numberOfChannels=channels, 
#                                        numberOfClasses=config.num_classes, 
#                                        beta=beta, 
#                                        threshold=threshold, 
#                                        numberOfSteps=numberOfSteps, 
#                                        expansion=config.expansion)
#         elif config.resnet_model == 18:
#             model = SpikeResNet18Model(numberOfChannels=channels, 
#                                        numberOfClasses=config.num_classes, 
#                                        beta=beta, 
#                                        threshold=threshold, 
#                                        numberOfSteps=numberOfSteps, 
#                                        expansion=config.expansion)
#         else:
#             print("Not defined")
#             return -1
        
#         model = model.to(device)
#         # Load weights
#         torch.cuda.empty_cache()

#         weightsName ='weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_L_'+str(config.loss)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed)+'.pth'        # Check weights file existence before loading
#         if not os.path.exists(weightsName):
#             print(f"Weights file not found: {weightsName}")
#             wN = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed)+'.pth'
#             if os.path.exists(wN):
#                 print(f"Found legacy weights file: {wN}, renaming to new format.")
#                 try:
#                     os.rename(wN, weightsName)
#                     print(f"Renamed legacy weights file {wN} -> {weightsName}")
#                 except Exception as e:
#                     print(f"Warning: failed to rename {wN} -> {weightsName}: {e}")
#             else:
#                 print(f"Weights file not found: {weightsName}")
#                 return -1, -1
#         # print(f"Loading weights from {weightsName}")
#         model.load_state_dict(torch.load(weightsName, weights_only=False))
#         model = model.to(device)
#         model.eval()

#         # print(f"Testing with expansion={config.expansion} ({'Population' if config.expansion > 1 else 'Standard'} coding)")
#         # print(f"Model has expansion attribute: {hasattr(model, 'expansion')}")
#         if hasattr(model, 'expansion'):
#             print(f"Model expansion value: {model.expansion}")
#         # print(f"Number of batches: {len(test_loader)}")

#         # Use same variables as your original test_accuracy_population
#         total = 0
#         acc_spikes = 0
#         acc_membrane = 0
        
#         # Convert to iterator like your original
#         test_iterator = iter(test_loader)
        
#         with torch.no_grad():
#             for data, targets in test_iterator:
#                 data = data.to(device)
#                 targets = targets.to(device)
                
#                 utils.reset(model)
                
#                 # Forward pass - get all three outputs
#                 spikes, _, membranes, _ = model(data, config.num_time_steps_extract)
                
#                 # Use same spike accuracy logic as fit_spike_full_train
#                 batch_size = data.size(0)
                
#                 acc_rate_spk = SF.accuracy_rate(
#                     spikes,
#                     targets,
#                     population_code=True,
#                     num_classes=model.numberOfClasses
#                 )
#                 batch_correct_spk = (acc_rate_spk * batch_size).item()
#                 acc_spikes += batch_correct_spk
                
#                 # Original membrane accuracy calculation
#                 # if hasattr(model, 'expansion') and model.expansion > 1:
#                 #     # Population coding mode - time average first, then population coding
#                 membranes_avg = membranes.mean(dim=0)  # [B, classes*expansion]
                
#                 acc_rate_mem = SF.accuracy_rate(
#                     membranes_avg.unsqueeze(0),  # Add time dimension [1, B, classes*expansion]
#                     targets, 
#                     population_code=True, 
#                     num_classes=model.numberOfClasses
#                 )
#                 batch_correct_mem = (acc_rate_mem * membranes_avg.size(0)).item()
#                 acc_membrane += batch_correct_mem
                
#                 total += batch_size
                
#                 # Progress display - same as your original
#                 spike_acc_pct = acc_spikes / total * 100
#                 mem_acc_pct = acc_membrane / total * 100
#                 print(f"\rTest accuracy on spikes: {spike_acc_pct:05.2f}%, "
#                       f"Test accuracy on membrane: {mem_acc_pct:05.2f}%, "
#                       f"progress: {total//config.batch_size}/{len(test_loader)}", 
#                       end='', flush=True)
        
#         print()  # New line
#         final_spike_acc = acc_spikes / total
#         final_mem_acc = acc_membrane / total
        
#         print(f"Final spike accuracy: {final_spike_acc*100:.2f}%")
#         print(f"Final membrane accuracy: {final_mem_acc*100:.2f}%")
        
#         del model 
#         torch.cuda.empty_cache()
#         return final_spike_acc*100, final_mem_acc*100
    
#     elif config.model_type == 'conv':
#         # Define model
#         if config.resnet_model == 4:
#             model = convNN4(numberOfChannels=channels, 
#                             numberOfClasses=config.num_classes, 
#                             feature_size=32, 
#                             expansion=config.expansion)
#         elif config.resnet_model == 10:
#             model = ResNet10(numberOfChannels=channels, 
#                                     numberOfClasses=config.num_classes, 
#                                     expan=config.expansion)
#         elif config.resnet_model == 18:
#             model = ResNet18(numberOfChannels=channels, 
#                                         numberOfClasses=config.num_classes, 
#                                         expan=config.expansion)
#         else:
#             print("Model not defined for the given ResNet configuration.")
#             return -1

#         # Load weights
#         weightsName = 'weights/conv/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_' + str(config.num_time_steps_train) + '_E_' + str(config.expansion) + '_A_' + str(config.auto_aug) + '_S_' + str(config.seed) + '.pth'
#         if not os.path.exists(weightsName):
#             print(f"Weights file not found: {weightsName}")
#             return -1

#         model.load_state_dict(torch.load(weightsName, weights_only=False))
#         model = model.to(device)
#         model.eval()

#         # Evaluate model
#         total_correct_membrane = 0
#         total_samples = 0
#         with torch.no_grad():
#             for batch, labels in test_loader:
#                 batch = batch.to(device)
#                 labels = labels.to(device)

#                 # Forward pass
#                 mem, _ = model(batch)

#                 # mem = membrane.mean(0)

#                 # Handle population coding if model.expansion > 1
#                 if hasattr(model, 'expansion') and model.expansion > 1:
#                     mem = mem.reshape(mem.shape[0], model.numberOfClasses, model.expansion).sum(dim=2)

#                 predicted = torch.argmax(mem, dim=1)
#                 correct = (predicted == labels).float()
#                 total_correct_membrane += correct.sum()
#                 total_samples += batch.size(0)

#         acc_membrane = total_correct_membrane / total_samples * 100
#         print(f"Inside test.py: {acc_membrane}")
#         return acc_membrane

    # else:
    #     # Existing logic for 'spike' case
    #     # For spiking neural network we need number of steps
    #     beta = 0.95
    #     threshold = 0.25

    #     # Define model
    #     if config.resnet_model == 1:
    #         model = spikeConvNN1(numberOfChannels=channels,
    #                              numberOfClasses=config.num_classes, 
    #                              beta=beta, 
    #                              threshold=threshold)
    #     elif config.resnet_model == 2:
    #         model = spikeConvNN2(numberOfChannels=channels, 
    #                              numberOfClasses=config.num_classes, 
    #                              beta=beta, 
    #                              threshold=threshold,
    #                              feature_size=feature_size)
    #     elif config.resnet_model == 4:
    #         model = spikeConvNN4(numberOfChannels=channels, 
    #                              numberOfClasses=config.num_classes, 
    #                              beta=beta, 
    #                              threshold=threshold, 
    #                              feature_size=feature_size, 
    #                              numberOfSteps=config.num_time_steps_train, 
    #                              expansion=config.expansion)
    #         model = model.to(device)
    #     elif config.resnet_model == 9:
    #         model = SpikeResNet9Model(numberOfChannels=channels, 
    #                                   numberOfClasses=config.num_classes, 
    #                                   beta=beta, 
    #                                   threshold=threshold)
    #     elif config.resnet_model == 10:
    #         model = SpikeResNet10Model(numberOfChannels=channels, 
    #                                     numberOfClasses=config.num_classes, 
    #                                     beta=beta, 
    #                                     threshold=threshold, 
    #                                     numberOfSteps=config.num_time_steps_train, 
    #                                     expansion=config.expansion)
    #     elif config.resnet_model == 18:
    #         model = SpikeResNet18Model(numberOfChannels=channels, 
    #                                    numberOfClasses=config.num_classes, 
    #                                    beta=beta, 
    #                                    threshold=threshold, 
    #                                    numberOfSteps=config.num_time_steps_train, 
    #                                    expansion=config.expansion)
    #     elif config.resnet_model == 20:
    #         model = SpikeResNet20Model(numberOfChannels=channels, 
    #                                    numberOfClasses=config.num_classes, 
    #                                    beta=beta, 
    #                                    threshold=threshold, 
    #                                    numberOfSteps=config.num_time_steps_train, 
    #                                    expansion=config.expansion)
    #     elif config.resnet_model == 21:
    #         model = spikeLinearNet1(numberOfChannels=channels, numberOfClasses=config.num_classes, beta=beta, threshold=threshold)
    #     else:
    #         print("Not defined")
    #         return -1
        
    #     # Load weights
    #     torch.cuda.empty_cache()

    #     weightsName = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_L_'+str(config.loss)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed)+'.pth'
    #     if not os.path.exists(weightsName):
    #         # try alternate filename without the loss token (legacy name)
    #         wN = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed)+'.pth'
    #         if os.path.exists(wN):
    #             print(f"Found legacy weights file: {wN}, renaming to new format.")
    #             try:
    #                 os.rename(wN, weightsName)
    #                 print(f"Renamed legacy weights file {wN} -> {weightsName}")
    #             except Exception as e:
    #                 print(f"Warning: failed to rename {wN} -> {weightsName}: {e}")
    #         else:
    #             print(f"Weights file not found: {weightsName}")
    #             return -1, -1
    #     # print(f"weightsName: {weightsName}")
    #     model.load_state_dict(torch.load(weightsName, weights_only=False))
    #     model = model.to(device)

    #     # print("Check test images.")
    #     testLen = 0
    #     i = 0
    #     # print(f"Number of batches: {len(test_loader)}")
    #     model.eval()
        
    #     total_correct_spikes = 0
    #     total_correct_membrane = 0
    #     total_samples = 0
    #     with torch.no_grad():
    #         for batch, labels in test_loader:
    #             testLen += len(batch)
    #             batch = batch.to(device)
    #             la = labels[0].item()
    #             labels = labels.to(device)
    #             # model.mem1 = final_membranes['mem1'].to(device)
    #             # model.mem2 = final_membranes['mem2'].to(device)
    #             # model.mem3 = final_membranes['mem3'].to(device)
    #             # model.mem4 = final_membranes['mem4'].to(device)
    #             # model.mem5 = final_membranes['mem5'].to(device)
    #             # Generate predictions/ forward pass
    #             spikes, _, membrane, _ = model(batch, config.num_time_steps_train)
    #             # print(f"spikes.max: {spikes.max()}")
    #             # l_spikes = loss_fn(spikes, labels)
    #             # testLossSpikes.append(l_spikes.item())
    #             # a1, a2 = model.accuracy_spike(model, numberOfSteps, batch, labels, device)
    #             # print(f"testLen: {testLen}, a1: {a1}, {a1.item()}, a2: {a2}, {a2.item()}")
    #             # testAcc[0, i] = a2.item()
    #             # testAccList.append(a1.item())
    #             # On spikes
    #             with torch.no_grad():
    #                 # Use SF.accuracy_count on the spikes from training forward pass
    #                 # acc_rate = SF.accuracy_rate(spikes, labels)
    #                 batch_size = batch.size(0)
    #                 acc = SF.accuracy_rate(spikes, labels) 
    #                 acc *= batch_size
    #                 total_correct_spikes += acc
    #                 total_samples += batch_size

    #             # On membrane
    #             mem = membrane.mean(0)
    #             predicted = torch.argmax(mem, dim=1)
    #             correct = (predicted == labels).float()
    #             total_correct_membrane += correct.sum()
    #             del batch, labels
    #             print(f"\rCurrent: {total_correct_spikes}, Test accuracy on spikes: {total_correct_spikes/total_samples*100:05.2f}, Test accuracy on membrane: {total_correct_membrane/total_samples*100:05.2f}, progress: {i+1}/{len(test_loader)}", end='', flush=True)
    #             i += 1

    #     print("\nDone.")
    #     acc_membrane = total_correct_membrane / total_samples * 100
    #     acc_spikes = total_correct_spikes / total_samples * 100

    #     del model 
    #     del train_loader, test_loader
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     return acc_spikes, acc_membrane

