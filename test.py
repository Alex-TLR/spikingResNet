from xml.parsers.expat import model
import numpy as np
import time 
from utils.Utils import Utils, distances_from_average_clusters, get_preds_from_probs_vector
from metrics.Metrics import Metrics, get_dist
from clustering.Clustering import Clustering
from models.spikeresnet import spikeConvNN1, spikeConvNN2, spikeConvNN4, SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model, SpikeResNet20Model  
from models.plain import spikeLinearNet1
import torch 
import snntorch.functional as SF 
import matplotlib.pyplot as plt
import torch.nn as nn
from snntorch import utils

from spikingjelly.clock_driven import neuron, surrogate, functional
from spikingjelly.clock_driven.model import sew_resnet

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

def test_accuracy(dataSet, modelType, batchSize, numberOfClasses, ResNetModel=None, num_steps=50, expansion=1, auto_aug=False):
    '''
    check accuracy of trained model on ID test data
    '''

    # Load database
    # print(f"dataSet: {dataSet}")
    dataset_train, dataset_test = Utils.load_data(dataSet)
    testSize = len(dataset_test)
    # print(f"Number of test set images is: {testSize}")

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, dataSet)

    # Load data
    train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, True)

    # Get the device
    device = Utils.get_device()
    # device = torch.device("cpu")

    if modelType == 'conv':
        pass

    elif modelType == 'spike':

        # For spiking neural network we need number of steps
        numberOfSteps = num_steps
        beta = 0.95
        threshold = 0.25

        # Define model
        if ResNetModel == 1:
            model = spikeConvNN1(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 2:
            model = spikeConvNN2(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 4:
            model = spikeConvNN4(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
            model = model.to(device)
        elif ResNetModel == 9:
            model = SpikeResNet9Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 10:
            model = SpikeResNet10Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold, numberOfSteps=numberOfSteps, expansion=expansion)
        elif ResNetModel == 18:
            # model = SpikeResNet18Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
            model = SpikeResNet18Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold, numberOfSteps=numberOfSteps, expansion=expansion)
        elif ResNetModel == 20:
            model = SpikeResNet20Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold, numberOfSteps=numberOfSteps, expansion=expansion)
        elif ResNetModel == 21:
            model = spikeLinearNet1(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 22:
            T = 4
            backend = 'torch'
            model = sew_resnet.multi_step_sew_resnet18(pretrained=False, progress=True, T=T, cnf='ADD', multi_step_neuron=neuron.MultiStepIFNode, v_threshold=1., surrogate_function=surrogate.ATan(), detach_reset=True, backend=backend, num_classes=10)
            # model = model.to(device)
        else:
            print("Not defined")
            return -1
        
        # Load weights
        torch.cuda.empty_cache()

        if ResNetModel != 22:


            weightsName = 'weights/spike/resnet' + str(ResNetModel) + '_weights_' + dataSet + '_T_'+str(numberOfSteps)+'_E_'+str(expansion)+'_A_'+str(auto_aug)+'.pth'
            print(f"weightsName: {weightsName}")
            model.load_state_dict(torch.load(weightsName, weights_only=False))
            # file = torch.load(weightsName)
            # model.load_state_dict(file["model"])
            model = model.to(device)

            # final_membranes = torch.load('final_membranes.pth')

            # Loss function
            # loss_fn = SF.ce_rate_loss()
            loss_fn = SF.ce_count_loss() 

            # print("Check test images.")
            testAcc = np.zeros((1,testSize))
            testLen = 0
            testAccList = []
            # testLossSpikes = []
            # testLossMembrane = []
            i = 0
            print(f"Number of batches: {len(test_loader)}")
            model.eval()
            
            total_correct_spikes = 0
            total_correct_membrane = 0
            total_samples = 0
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
                    spikes, feat, membrane = model(batch, numberOfSteps)
                    # l_spikes = loss_fn(spikes, labels)
                    # testLossSpikes.append(l_spikes.item())
                    # a1, a2 = model.accuracy_spike(model, numberOfSteps, batch, labels, device)
                    # print(f"testLen: {testLen}, a1: {a1}, {a1.item()}, a2: {a2}, {a2.item()}")
                    # testAcc[0, i] = a2.item()
                    # testAccList.append(a1.item())
                    # On spikes
                    with torch.no_grad():
                        # Use SF.accuracy_count on the spikes from training forward pass
                        # acc_rate = SF.accuracy_rate(spikes, labels)
                        batch_size = batch.size(0)
                        acc = SF.accuracy_rate(spikes, labels) * spikes.size(1)
                        total_correct_spikes += acc
                        total_samples += batch_size

                    # On membrane
                    mem = membrane.mean(0)
                    predicted = torch.argmax(mem, dim=1)
                    correct = (predicted == labels).float()
                    total_correct_membrane += correct.sum()
                    del batch, labels
                    # print(f"\rtestAcc[{i}]: {testAcc[0, i]}, label: {la}, count: {(np.sum(testAcc[0, :])/(i+1)):.2f}, progress: {i}/{len(test_loader)}", end='', flush=True)
                    print(f"\rCurrent: {total_correct_spikes}, Test accuracy on spikes: {total_correct_spikes/total_samples*100:05.2f}, Test accuracy on membrane: {total_correct_membrane/total_samples*100:05.2f}, progress: {i+1}/{len(test_loader)}", end='', flush=True)
                    i += 1

            print("\nDone.")
            # Test stats
            # meanA1 = np.sum(testAcc) / testSize
            # meanA2 = sum(testAccList) / len(testAccList)
            # meanL = sum(testLoss) / len(testLoss)
            # print(f'\nTest loss is {meanL:.2f}. Test accuracy1 is {meanA1*100:.2f}. Test accuracy2 is {meanA2*100:.2f}.')

            return None 
    
        else:
            weightsName = 'weights/spike/' + 'resnet' + str(ResNetModel) + '_weights_' + dataSet + '.pth'
            model.load_state_dict(torch.load(weightsName, weights_only=False))
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            testAccList1 = []
            testAccList5 = []
            testAcc = np.zeros((1,testSize))
            with torch.no_grad():
                for batch, labels in test_loader:
                    image = batch.to(device, non_blocking=True)
                    target = labels.to(device, non_blocking=True)
                    output = model(image)
                    output = output.mean(dim=0)
                    loss = criterion(output, target)
                    functional.reset_net(model)

                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    testAccList1.append(acc1.item())
                    testAccList5.append(acc5.item())
                    print(f'Test accuracy1: {acc1.item():.2f}%, Test accuracy5: {acc5.item():.2f}%, Loss: {loss.item():.2f}')

            print(f"Final results: Test accuracy1: {np.mean(testAccList1):.2f}%, Test accuracy5: {np.mean(testAccList5):.2f}%")
            return None


def test_accuracy_population(dataSet, model, modelType, batchSize, numberOfClasses, ResNetModel, expansion=1, auto_aug=False):
    dataset_train, dataset_test = Utils.load_data(dataSet)
    testSize = len(dataset_test)
    channels, rows, cols = Utils.get_image_size(dataset_train, dataSet)
    train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, True)
    device = Utils.get_device()

    # Get image size based on dataset
    if dataSet in ['CIFAR10', 'CIFAR100']:
        feature_size = 32
    elif dataSet in ['MNIST', 'FMNIST', 'KMNIST']:
        feature_size = 28
    else:
        feature_size = 28

    if modelType == 'spike':
        numberOfSteps = 4  # This parameter exists but isn't used in testing
        beta = 0.95
        threshold = 0.25
        
        model = model.to(device)
        model.eval()
        
        total = 0
        acc_spikes = 0
        acc_membrane = 0
        
        test_iterator = iter(test_loader)
        
        with torch.no_grad():
            for data, targets in test_iterator:
                data = data.to(device)
                targets = targets.to(device)
                
                # Only reset - no manual time steps
                utils.reset(model)
                
                # Single forward call (BPTT handles time internally)
                spk_rec, mem_rec = model(data)  # [B, 500] if single time step model
                
                
                # For spikes - population coding
                acc_rate_spk = SF.accuracy_rate(
                    spk_rec.unsqueeze(0),  # Add time dimension [1, B, 500]
                    targets, 
                    population_code=True, 
                    num_classes=numberOfClasses
                ) * spk_rec.size(0)  # Multiply by batch size to get count
                
                acc_spikes += acc_rate_spk
                
                acc_rate_mem = SF.accuracy_rate(
                    mem_rec.unsqueeze(0),  # Add time dimension [1, B, 500]
                    targets, 
                    population_code=True, 
                    num_classes=numberOfClasses
                ) * mem_rec.size(0)  # Multiply by batch size to get count
                
                acc_membrane += acc_rate_mem
                total += spk_rec.size(0)  # Add batch size
                
                # Progress display
                spike_acc_pct = acc_spikes / total * 100
                mem_acc_pct = acc_membrane / total * 100
                print(f"\rTest accuracy on spikes: {spike_acc_pct:05.2f}%, "
                      f"Test accuracy on membrane: {mem_acc_pct:05.2f}%, "
                      f"progress: {total//batchSize}/{len(test_loader)}", 
                      end='', flush=True)
        
        print()  # New line
        final_spike_acc = acc_spikes / total
        final_mem_acc = acc_membrane / total
        
        print(f"Final spike accuracy: {final_spike_acc*100:.2f}%")
        print(f"Final membrane accuracy: {final_mem_acc*100:.2f}%")
        
        return final_spike_acc


def test_accuracy_population_2(dataSet, modelType, batchSize, numberOfClasses,  ResNetModel, expansion=50, auto_aug=False):
    '''
    check accuracy of trained model on ID test data when population coding is used with my spatio-temporal propagation
    '''
    # Load database
    dataset_train, dataset_test = Utils.load_data(dataSet)
    testSize = len(dataset_test)

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, dataSet)

    # Load data
    train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, True)

    # Get the device
    device = Utils.get_device()

    # Get image size based on dataset
    if dataSet in ['CIFAR10', 'CIFAR100']:
        feature_size = 32
    elif dataSet in ['MNIST', 'FMNIST', 'KMNIST']:
        feature_size = 28
    else:
        feature_size = 28

    if modelType == 'spike':
        # For spiking neural network we need number of steps
        numberOfSteps = 1
        beta = 0.95
        threshold = 0.25

        # Define model with expansion parameter
        if ResNetModel == 2:
            model = spikeConvNN2(
                numberOfChannels=channels, 
                numberOfClasses=numberOfClasses, 
                beta=beta, 
                threshold=threshold, 
                expansion=expansion, 
                feature_size=feature_size
            )
        # elif ResNetModel == 4:
        #     print("This model is adapted for BPTT only.")
        #     return -1
        elif ResNetModel == 10:
            model = SpikeResNet10Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold, numberOfSteps=numberOfSteps, expansion=expansion)
            model = model.to(device)
        elif ResNetModel == 18:
            model = SpikeResNet18Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold, numberOfSteps=numberOfSteps, expansion=expansion)
            model = model.to(device)
        else:
            print("Not defined")
            return -1
        
        # Load weights
        torch.cuda.empty_cache()
        
        weightsName ='weights/spike/resnet' + str(ResNetModel) + '_weights_' + dataSet + '_T_'+str(numberOfSteps)+'_E_'+str(expansion)+'_A_'+str(auto_aug)+'.pth'

        model.load_state_dict(torch.load(weightsName, weights_only=False))
        model = model.to(device)
        model.eval()

        print(f"Testing with expansion={expansion} ({'Population' if expansion > 1 else 'Standard'} coding)")
        print(f"Model has expansion attribute: {hasattr(model, 'expansion')}")
        if hasattr(model, 'expansion'):
            print(f"Model expansion value: {model.expansion}")
        print(f"Number of batches: {len(test_loader)}")

        # Use same variables as your original test_accuracy_population
        total = 0
        acc_spikes = 0
        acc_membrane = 0
        
        # Convert to iterator like your original
        test_iterator = iter(test_loader)
        
        with torch.no_grad():
            for data, targets in test_iterator:
                data = data.to(device)
                targets = targets.to(device)
                
                utils.reset(model)
                
                # Forward pass - get all three outputs
                spikes, features, membranes = model(data, numberOfSteps)
                
                # Use same spike accuracy logic as fit_spike_full_train
                batch_size = data.size(0)
                
                # Check if model uses population coding (same logic as training)
                if hasattr(model, 'expansion') and model.expansion > 1:
                    # Population coding accuracy for spikes - same as fit_spike_full_train
                    acc_rate_spk = SF.accuracy_rate(
                        spikes, 
                        targets, 
                        population_code=True, 
                        num_classes=model.numberOfClasses
                    )
                    batch_correct_spk = (acc_rate_spk * batch_size).item()
                    
                else:
                    # Standard coding accuracy for spikes - same as fit_spike_full_train
                    acc_rate_spk = SF.accuracy_rate(spikes, targets)
                    batch_correct_spk = (acc_rate_spk * batch_size).item()
                
                acc_spikes += batch_correct_spk
                
                # Original membrane accuracy calculation
                if hasattr(model, 'expansion') and model.expansion > 1:
                    # Population coding mode - time average first, then population coding
                    membranes_avg = membranes.mean(dim=0)  # [B, classes*expansion]
                    
                    acc_rate_mem = SF.accuracy_rate(
                        membranes_avg.unsqueeze(0),  # Add time dimension [1, B, classes*expansion]
                        targets, 
                        population_code=True, 
                        num_classes=numberOfClasses
                    )
                    batch_correct_mem = (acc_rate_mem * membranes_avg.size(0)).item()
                    acc_membrane += batch_correct_mem
                    
                else:
                    # Standard coding mode
                    mem_avg = membranes.mean(0)  # [B, classes]
                    predicted = torch.argmax(mem_avg, dim=1)
                    correct = (predicted == targets).float().sum()
                    acc_membrane += correct.item()
                
                total += batch_size
                
                # Progress display - same as your original
                spike_acc_pct = acc_spikes / total * 100
                mem_acc_pct = acc_membrane / total * 100
                print(f"\rTest accuracy on spikes: {spike_acc_pct:05.2f}%, "
                      f"Test accuracy on membrane: {mem_acc_pct:05.2f}%, "
                      f"progress: {total//batchSize}/{len(test_loader)}", 
                      end='', flush=True)
        
        print()  # New line
        final_spike_acc = acc_spikes / total
        final_mem_acc = acc_membrane / total
        
        print(f"Final spike accuracy: {final_spike_acc*100:.2f}%")
        print(f"Final membrane accuracy: {final_mem_acc*100:.2f}%")
        
        return None


def test_metrics(case, nameID):
    '''
    case:       for example case 01 is '01'
    nameID:     name of the In Distribution features, example 'MNIST'
    '''
    if nameID == 'MNIST':
        namesOOD = ['FMNIST', 'KMNIST', 'Letters']
        suffixID = '-on_mnist'
    elif nameID == 'FMNIST':
        namesOOD = ['MNIST', 'KMNIST', 'Letters']   
        suffixID = '-on_fmnist'
    elif nameID == 'KMNIST':
        namesOOD = ['MNIST', 'FMNIST', 'Letters']
    elif nameID == 'Letters':
        namesOOD = ['MNIST', 'FMNIST', 'KMNIST']
        suffixID = '-on_letters'
    elif nameID == 'CIFAR10':
        namesOOD = ['SVHN', 'Food101']
        suffixID = '-on_cifar10'
    elif nameID == 'SVHN':
        namesOOD = ['CIFAR10', 'Food101']
        suffixID = '-on_svhn'

    # methods = ['MSP', 'NCM', 'KNN', 'NNDR', 'MD']
    methods = ['NCM', 'MD', 'KNN', 'FKM', 'CKM']
    # methods = ['KMEANS-Full' , 'KMEANS-Full2']
    stats = np.zeros((len(namesOOD), len(methods)*3), dtype=np.float64)
    IDpath = 'features/spike/case_' + case + '/' + nameID + suffixID + '.npz'

    ID = np.load(IDpath)
    # print(f"\nIDpath: {IDpath}")
    ID_spik_train = ID['arr0']  # In-Distribution training set spikes
    ID_feat_train = ID['arr1']  # In-Distribution training set features
    ID_prob_train = ID['arr2']  # In-Distribution training set outputs (usually with no softmax applied)
    ID_tags_train = ID['arr3']  # In-Distribution training set labels
    ID_spik_test  = ID['arr7']  # In-Distribution test set spikes
    ID_feat_test  = ID['arr4']  # In-Distribution test set features
    ID_prob_test  = ID['arr5']  # In-Distribution test set outputs (usually with no softmax applied)
    # ID_tags_test  = ID['arr6']  # In-Distribution test set labels
    number_classes = ID_prob_train.shape[1]
    # clusters = Clustering.clustering_1(ID_feat_train, ID_tags_train, number_classes)
    # print(f"Clustering ID base: {clusters}")

    for i in range(len(namesOOD)):
        OODpath = 'features/spike/case_' + case + '/' + namesOOD[i] + suffixID + '.npz'
        # print(f"OODpath: {OODpath}")
        print(f"OOD is: {namesOOD[i]}")
        OOD = np.load(OODpath)
        OOD_spik_train = OOD['arr0']  # Out-of-Distribution training set spikes
        OOD_feat_train = OOD['arr1']  # Out-of-Distribution training set features
        OOD_prob_train = OOD['arr2']  # Out-of-Distribution training set outputs (usually with no softmax applied)
        OOD_tags_train = OOD['arr3']  # Out-of-Distribution training set labels
        OOD_spik_test  = OOD['arr7']  # Out-of-Distribution test set spikes
        OOD_feat_test  = OOD['arr4']  # Out-of-Distribution test set features
        OOD_prob_test  = OOD['arr5']  # Out-of-Distribution test set outputs (usually with no softmax applied)
        OOD_tags_test  = OOD['arr6']  # Out-of-Distribution test set labels

        # print(f"OOD_prob_train.shape: {OOD_prob_train.shape}, OOD_prob_test.shape: {OOD_prob_test.shape}")
        # print(f"ID_feat_train.shape: {ID_feat_train.shape}, ID_feat_test.shape: {ID_feat_test.shape}, OOD_feat_train.shape: {OOD_feat_train.shape}")

        # Methodology includes the IN/OOD classification where ID are positive samples taken from ID_test_set
        # and OOD are negative samples taken from OOD_train_set (or maybe OOD_train_set + OOD_test_set)

        for j in range(len(methods)):

            # iterate trought each of classification methods and calculate the metrics
            # This is the baseline method that uses the outputs of the network number_of_classes-D 
            # if methods[j] == 'MSP':
            #     print("MSP")
            #     start_time = time.time() 
            #     ID_labels   = np.ones((len(ID_prob_test)))
            #     OOD_labels  = np.zeros((len(OOD_prob_train)))
            #     test_labels = np.concatenate((ID_labels, OOD_labels))
            #     # print(f"test_labels: {test_labels.shape}")
            #     threshold = 0

            #     # Calculates the distances between row-wise max probabilities and 0,
            #     # only to check the acctual max of probabilities (predictions are not
            #     # important in this step). 
            #     _ , ID_distances = Metrics.MSP(ID_prob_test, threshold)
            #     _, OOD_distances = Metrics.MSP(OOD_prob_train, threshold)
            #     # OOD_prob = np.concatenate((OOD_prob_train, OOD_prob_test), axis=0)
            #     test_distances = np.concatenate((ID_distances, OOD_distances))
            #     # plot test_label and test_distances

            #     # plt.figure(figsize=(10, 6))
            #     # plt.plot(test_distances[0:20000], alpha=0.5, label="Distances")
            #     # plt.plot(test_labels[0:20000], linewidth=2, label="Labels")
            #     # plt.title("MSP")
            #     # plt.xlabel("Sample")
            #     # plt.ylabel("Distance")
            #     # plt.grid(True)
            #     # plt.legend()
            #     # plt.show()


            #     _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
            #     threshold = threshold_tpr95
            #     ID_predictions, _  = Metrics.MSP(ID_prob_test, threshold)
            #     OOD_predictions, _ = Metrics.MSP(OOD_prob_train, threshold)
            #     test_predictions = np.concatenate((ID_predictions, OOD_predictions))
            #     auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
            #     # print(f"threshold_tpr95: {threshold_tpr95}")
            #     print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
            #     stats[i, j] = auroc 
            #     stats[i, len(methods) + j] = aupr
            #     stats[i, len(methods)*2 + j] = fpr95 
            #     end_time = time.time()  # Record end time
            #     execution_time = end_time - start_time  # Calculate execution time
            #     print(f"MSP Execution time: {execution_time:.4f} seconds.\n")

            if methods[j] == 'MSP':
                # MSP is the maximum softmax probability method
                # this method is optimized
                print("MSP")
                start_time = time.time() 
                ID_labels   = np.ones((len(ID_prob_test)))
                OOD_labels  = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))

                # Calculates softmax probabilities
                ID_distances = np.max(Metrics.softmax(ID_prob_test), axis=1)
                OOD_distances = np.max(Metrics.softmax(OOD_prob_train), axis=1)
                test_distances = np.concatenate((ID_distances, OOD_distances))

                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95

                # Make predictions based on distances and threshold
                ID_predictions = (ID_distances > threshold).astype(np.int32)
                OOD_predictions = (OOD_distances > threshold).astype(np.int32)

                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"MSP Execution time: {execution_time:.4f} seconds.\n")

            elif methods[j] == 'SD':
                # this method needs revision
                print("Spike distance")
                print("Keep output spike pattern, numOfClasses-D")
                start_time = time.time() 
                number_classes = ID_prob_train.shape[1]
                ID_labels   = np.ones((len(ID_prob_test)))
                OOD_labels  = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                # print(f"test_labels: {test_labels.shape}")
                threshold = 0

                # Calculates the distances between row-wise max probabilities and 0,
                # only to check the acctual max of probabilities (predictions are not
                # important in this step). 
                _ , ID_distances = Metrics.SD(ID_spik_train, ID_tags_train, ID_spik_test, threshold, number_classes)
                _, OOD_distances = Metrics.SD(ID_spik_train, ID_tags_train, OOD_spik_train, threshold, number_classes)
                # OOD_prob = np.concatenate((OOD_prob_train, OOD_prob_test), axis=0)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                # plot test_label and test_distances

                # plt.figure(figsize=(10, 6))
                # plt.plot(test_distances[0:20000], alpha=0.5, label="Distances")
                # plt.plot(test_labels[0:20000], linewidth=2, label="Labels")
                # plt.title("MSP")
                # plt.xlabel("Sample")
                # plt.ylabel("Distance")
                # plt.grid(True)
                # plt.legend()
                # plt.show()


                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                ID_predictions, _  = Metrics.SD(ID_spik_train, ID_tags_train, ID_spik_test, threshold, number_classes)
                OOD_predictions, _ = Metrics.SD(ID_spik_train, ID_tags_train, OOD_spik_train, threshold, number_classes)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                # print(f"threshold_tpr95: {threshold_tpr95}")
                print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"SD (spike distance) Execution time: {execution_time:.4f} seconds.\n")

            elif methods[j] == 'NCM':
                print(f"NCM on {namesOOD[i]}")
                number_classes = ID_prob_train.shape[1]
                test_labels, test_predictions, test_distances = Metrics.NCM(ID_feat_train, ID_tags_train, ID_feat_test, OOD_feat_test, number_classes)
                # test_labels, test_predictions, test_distances = Metrics.NCM(ID_feat_train, ID_tags_train, ID_feat_test, OOD_feat_test, number_classes)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                # print(f"NCM Done.\n")

            elif methods[j] == 'MD':
                print(f"MD on {namesOOD[i]}")
                number_classes = ID_prob_train.shape[1]
                number_features = ID_feat_train.shape[1]
                test_labels, test_predictions, test_distances = Metrics.MD(ID_feat_train, ID_tags_train, ID_feat_test, OOD_feat_test, number_classes, number_features)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                # print(f"MD Done.")

            elif methods[j] == 'KNN':
                print(f"KNN on {namesOOD[i]}")
                number_neighbors = 10
                test_labels, test_predictions, test_distances = Metrics.KNN(ID_feat_train, ID_tags_train, ID_feat_test, OOD_feat_test, number_neighbors)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 

            elif methods[j] == 'FKM':
                print(f"K-MEANS Full Clustering on {namesOOD[i]}")
                number_neighbors = 100
                test_labels, test_predictions, test_distances = Metrics.FKM(ID_feat_train, ID_feat_test, OOD_feat_test, number_neighbors)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                # print(f"KMEANS Full Execution done")

            elif methods[j] == 'CKM':
                print(f"K-MEANS Clustering per Class on {namesOOD[i]}")
                number_classes = ID_prob_train.shape[1]
                ncpc = 5
                test_labels, test_predictions, test_distances = Metrics.CKM(ID_feat_train, ID_tags_train, ID_feat_test, OOD_feat_test, number_classes, ncpc)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                
                print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                # print(f"CKM Done.")

            elif methods[j] == 'NNDR':
                # needs revision
                start_time = time.time()
                number_neighbors = 11000 
                ID_labels = np.ones((len(ID_prob_test)))
                OOD_labels = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _, ID_distances  = Metrics.NNDR(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
                _, OOD_distances = Metrics.NNDR(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_neighbors)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                ID_predictions, _  = Metrics.NNDR(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
                OOD_predictions, _ = Metrics.NNDR(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_neighbors)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                print(f"threshold_tpr95: {threshold_tpr95}")
                print(f"True positive rate: {tpr95}, False positive rate {fpr95}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"NNDR Execution time: {execution_time:.4f} seconds")

            # elif methods[j] == 'AGGLO':
            #     # needs revision
            #     start_time = time.time()
            #     number_classes = ID_prob_train.shape[1]
            #     ID_labels = np.ones((len(ID_prob_test)))
            #     OOD_labels = np.zeros((len(OOD_prob_train)))
            #     test_labels = np.concatenate((ID_labels, OOD_labels))
            #     threshold = 0

            #     # total = 0
            #     # for ii in range(number_classes):
            #     #     samples = ID_feat_train[ID_tags_train == ii]
            #     #     print(f"number of samples of class {ii} is {samples.shape}")
            #     #     total += samples.shape[0]
            #     # print(f"total: {total}")


            #     # Find clusters (centroids) per class
            #     clusterName = 'clustering/cluster_2.npy'
            #     if (Utils.does_file_exists(clusterName)):
            #         predictions = get_preds_from_probs_vector(ID_prob_train) 
            #         clusters = Clustering.clustering_2(ID_feat_train, predictions, number_classes)
            #         print(f"Cluster per class: {len(clusters)}")
            #         np.save(clusterName, clusters, allow_pickle=True)
            #     else:
            #         # Load cluster
            #         clusters = np.load(clusterName, allow_pickle=True)

            #     # Calculate cluster average clusters (centroids) for train features
            #     averagePerClass = []
            #     for ii in range(number_classes):
            #         averageCluster = []
            #         for cluster_index in np.unique(clusters[ii].labels_):
            #             averageCluster.append(np.median(ID_feat_train[np.where(clusters[ii].labels_ == cluster_index)[0]], axis=0))
            #         averagePerClass.append(np.array(averageCluster))

            #     # Compute distance for each sample from centroids according to predicted class
            #     ID_train_distances = distances_from_average_clusters(
            #         ID_feat_train, ID_spik_train, averagePerClass, ID_prob_train)
            #     ID_distances = distances_from_average_clusters(
            #         ID_feat_test, ID_spik_test, averagePerClass, ID_prob_test)
            #     OOD_distances = distances_from_average_clusters(
            #         OOD_feat_train, OOD_spik_train, averagePerClass, OOD_prob_test)
                
            #     # averagePerClass = np.row_stack(averagePerClass)

            #     # Compute thresholds for each class based on ID_train_distances
            #     thresholds = compute_thresholds(ID_train_distances)
            #     print(thresholds[:, 94])

            #     precision, tpr_values, fpr_values = compute_precision_tpr_fpr_for_test_and_ood(
            #         ID_distances, OOD_distances, thresholds)
            #     # print(precision, tpr_values, fpr_values)
            #     # Appending that when FPR = 1 the TPR is also 1:
            #     tpr_values_auroc = np.append(tpr_values, 1)
            #     fpr_values_auroc = np.append(fpr_values, 1)
            #     # Metrics
            #     auroc = round(np.trapz(tpr_values_auroc,
            #                   fpr_values_auroc), 2)
            #     aupr = round(np.trapz(precision, tpr_values), 2)
            #     fpr95 = round(fpr_values_auroc[95], 2)
            #     fpr80 = round(fpr_values_auroc[80], 2)
            #     print(f"auroc: {auroc}, aupr: {aupr}, fpr95: {fpr95}")

            #     averagePerClass = np.row_stack(averagePerClass)


            #     _, ID_distances  = Metrics.AGGLO(ID_feat_test, averagePerClass, threshold, number_classes)
            #     _, OOD_distances = Metrics.AGGLO(OOD_feat_train, averagePerClass, threshold, number_classes)
            #     test_distances = np.concatenate((ID_distances, OOD_distances))
            #     _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
            #     threshold = threshold_tpr95
            #     ID_predictions, _  = Metrics.AGGLO(ID_feat_test, averagePerClass, threshold, number_classes)
            #     OOD_predictions, _ = Metrics.AGGLO(OOD_feat_train, averagePerClass, threshold, number_classes)
            #     test_predictions = np.concatenate((ID_predictions, OOD_predictions))
            #     auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
            #     print(f"threshold_tpr95: {threshold_tpr95}")
            #     print(f"True positive rate: {tpr95}, False positive rate {fpr95}")
            #     stats[i, j] = auroc 
            #     stats[i, len(methods) + j] = aupr
            #     stats[i, len(methods)*2 + j] = fpr95 
            #     end_time = time.time()  # Record end time
            #     execution_time = end_time - start_time  # Calculate execution time
            #     print(f"AGGLO Execution time: {execution_time:.4f} seconds")

            elif methods[j] == 'DBSCAN':
                print("DBSCAN")
                start_time = time.time()
                number_classes = ID_prob_train.shape[1]
                ID_labels = np.ones((len(ID_prob_test)))
                OOD_labels = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _, ID_distances  = Metrics.DBSCAN(ID_feat_train, ID_feat_test, threshold)
                _, OOD_distances = Metrics.DBSCAN(ID_feat_train, OOD_feat_train, threshold)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                print(f"threshold_tpr95: {threshold_tpr95}")

                ID_predictions, _  = Metrics.DBSCAN(ID_feat_train, ID_feat_test, threshold)
                OOD_predictions, _ = Metrics.DBSCAN(ID_feat_train, OOD_feat_train, threshold)

                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                # print(f"threshold_tpr95: {threshold_tpr95}")
                print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"DBSCAN Execution time: {execution_time:.4f} seconds")

            else:
                pass

        del OOD, OOD_feat_train, OOD_prob_train, OOD_prob_test
        print()

    return stats
