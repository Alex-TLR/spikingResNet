import torch
from utils.Utils import Utils
from torch.utils.data import DataLoader
from models.resnet import ResNet9Model 
from models.spikeresnet import spikeConvNN1, spikeConvNN2, spikeConvNN4, SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model, SpikeResNet20Model
from models.plain import spikeLinearNet1
import numpy as np
import matplotlib.pyplot as plt
from metrics.Metrics import Metrics


def feature_extraction_conv(dataSet):
    '''
    Feature extraction out of regular ResNet9
    '''
    dataset_train, dataset_test = Utils.load_data(dataSet)

    device = Utils.get_device()

    trainDataSize = len(dataset_train)
    # print("Train data size: ", trainDataSize)
    testDataSize = len(dataset_test)
    # print("Test data size: ", testDataSize)

    # Number of classes
    numberOfClasses = 10

    # Batch size
    batchSize = 50

    # There is not validation data
    train_loader = DataLoader(dataset_train, batchSize)
    test_loader = DataLoader(dataset_test, batchSize)

    # Define model
    model = ResNet9Model(numberOfChannels=1, numberOfClasses=numberOfClasses)
    # print(model)
    # Load weights
    weightsName = 'weights/conv/' + 'resnet9_weights_' + dataSet + '.pth'
    model.load_state_dict(torch.load(weightsName, weights_only=True))
    model = model.to(device)

    # extractor = CustomResNetConv(num_classes=numberOfClasses, network=model) 
    # print(extractor)
    # extractor = extractor.to(device)

    Feat_train = np.zeros((trainDataSize, 512))
    Prob_train = np.zeros((trainDataSize, 10))
    Tags_train = []
    i = 0    
    for batch, labels in train_loader:
        batch = batch.to(device)
        labels = labels.to(device)
        startIndex = i*batchSize
        endIndex = startIndex + batchSize

        probs, feats = model(batch)
        labels = labels.cpu().detach().numpy()
        feats = feats.cpu().detach().numpy()
        probs = probs.cpu().detach().numpy()
        print(f'feats.shape is {feats.shape} and probs.shape is {probs.shape}')
        Feat_train[startIndex:endIndex, :] = feats
        Prob_train[startIndex:endIndex, :] = probs
        Tags_train.append(labels.flatten())
        del batch, labels, probs
        i += 1

    # Tags_train = np.array(Tags_train)
    Tags_train = np.concatenate(Tags_train)
    print("Tags_train shape is ", Tags_train.shape)

    Feat_test = np.zeros((testDataSize, 512))
    Prob_test = np.zeros((testDataSize, 10))
    Tags_test = []
    i = 0
    for batch, labels in test_loader:
        batch = batch.to(device)
        labels = labels.to(device)
        startIndex = i*batchSize
        endIndex = startIndex + batchSize

        probs, feats = model(batch)
        labels = labels.cpu().detach().numpy()
        Feat_test[startIndex:endIndex, :] = feats.cpu().detach().numpy()
        Prob_test[startIndex:endIndex, :] = probs.cpu().detach().numpy()
        Tags_test.append(labels.flatten())
        del batch, labels, probs
        i += 1

    # Tags_test = np.array(Tags_test)
    Tags_test = np.concatenate(Tags_test)
    print("Tags_test shape is ", Tags_test.shape)

    fileName = 'features/case_04/' + dataSet + '-on_mnist' + '.npz'
    np.savez(fileName, arr1=Feat_train, arr2=Prob_train, arr3=Tags_train, arr4=Feat_test, arr5=Prob_test, arr6=Tags_test)

    return None


# def feature_extraction_spike(dataSet_ID, dataSet_feat, ResNetModel, case, numOfClasses, batchSize, expansion=1, auto_aug=False):
def feature_extraction_spike(config):
    '''
    Spiking models only
    dataSet:        the data set from which we extract feature
                    at the moment, features are extracted only on MNIST-trained network.
    dataSet_ID:     in-definition dataset
    dataSet_feat:   dataset for feature extraction
    ResNetModel:    select resnet model type (resnet10, resnet18)
    numOfClasses:   number of classes
    numOfChannels:  number of input channels
    '''

    # print(f"Feature length is: {len(config.dataset_feat)}")
    for i in range(len(config.dataset_feat)):
        print(f"Extracting features for {config.dataset_feat[i]}")
        dataset_train, dataset_test = Utils.load_data(config.dataset_feat[i])

        # Get image size
        channels, rows, cols = Utils.get_image_size(dataset_train, config.dataset_feat[i])
        # print(f"Image size: {channels, rows, cols}")

        device = Utils.get_device()

        # Get image size based on dataset
        if config.dataset_ID in ['CIFAR10', 'CIFAR100']:
            feature_size = 32
        elif config.dataset_ID in ['MNIST', 'FMNIST', 'KMNIST']:
            feature_size = 28
        else:
            feature_size = 28

        trainDataSize = len(dataset_train)
        # print("Train data size: ", trainDataSize)
        testDataSize = len(dataset_test)
        # print("Test data size: ", testDataSize)
        # print(f"Network models is {ResNetModel}")

        # Parameter of the LIF neuron
        beta = 0.95

        # Threshold
        threshold = 0.25

        # Load OoD data
        train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_feat[i], True)

        # Loss function
        # loss_fn = SF.ce_rate_loss()
        # loss_fn = SF.ce_count_loss() 

        # Define 
        if config.resnet_model == 1:
            model = spikeConvNN1(numberOfChannels=channels, 
                                 numberOfClasses=config.num_classes, 
                                 beta=beta, 
                                 threshold=threshold)
            featSize = 256
        elif config.resnet_model == 2:
            model = spikeConvNN2(numberOfChannels=channels, 
                                 numberOfClasses=config.num_classes,
                                 beta=beta, threshold=threshold, 
                                 feature_size=feature_size, 
                                 expansion=config.expansion)    
            featSize = 300
        elif config.resnet_model == 4:
            model = spikeConvNN4(numberOfChannels=channels, 
                                 numberOfClasses=config.num_classes, 
                                 beta=beta, threshold=threshold, 
                                 feature_size=feature_size, 
                                 numberOfSteps=config.num_time_steps_extract, 
                                 expansion=config.expansion)
            featSize = 256
        elif config.resnet_model == 9:
            model = SpikeResNet9Model(numberOfChannels=channels, 
                                      numberOfClasses=config.num_classes, 
                                      beta=beta, 
                                      threshold=threshold)
            featSize = 512
        elif config.resnet_model == 10:
            model = SpikeResNet10Model(numberOfChannels=channels, 
                                       numberOfClasses=config.num_classes, 
                                       beta=beta, 
                                       threshold=threshold, 
                                       numberOfSteps=config.num_time_steps_extract, 
                                       expansion=config.expansion)
            featSize = 512
        elif config.resnet_model == 18:
            model = SpikeResNet18Model(numberOfChannels=channels, 
                                       numberOfClasses=config.num_classes, 
                                       beta=beta, 
                                       threshold=threshold, 
                                       numberOfSteps=config.num_time_steps_extract, 
                                       expansion=config.expansion)
            featSize = 512
        elif config.resnet_model == 20:
            model = SpikeResNet20Model(numberOfChannels=channels, 
                                       numberOfClasses=config.num_classes, 
                                       beta=beta, 
                                       threshold=threshold, 
                                       numberOfSteps=config.num_time_steps_extract, 
                                       expansion=config.expansion)
            featSize = 512
        elif config.resnet_model == 21:
            model = spikeLinearNet1(numberOfChannels=channels, 
                                     numberOfClasses=config.num_classes, 
                                     beta=beta, 
                                     threshold=threshold)
            featSize = 300
        else:
            print("Feature: Not defined")
            return -1
        
        # Load weights
        # Loading the weights for the ID-trained network
        weightsName = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_A_'+str(config.auto_aug)+'.pth'
        # weightsName = 'weights/spike/resnet' + str(ResNetModel) + '_weights_' + dataSet_ID + '.pth'
        model.load_state_dict(torch.load(weightsName, weights_only=True))
        print(f"weights file {weightsName}")
        model = model.to(device)
        # model.reset_mem(batchSize, device)
        # final_membranes = torch.load('final_membranes.pth')

        fileName = 'features/spike/case_' + config.case + '/' + config.dataset_feat[i] + '-on_' + config.dataset_ID.lower() + '.npz'
        # print(fileName)

        if (Utils.does_file_exists(fileName)):

            torch.cuda.empty_cache()
            Spik_train = np.zeros((trainDataSize, config.num_classes))
            Feat_train = np.zeros((trainDataSize, featSize))
            Prob_train = np.zeros((trainDataSize, config.num_classes))
            Volt_train = np.zeros((trainDataSize, featSize))
            Tags_train = []
            i = 0   
            with torch.no_grad(): 
                for batch, labels in train_loader:
                    # if i == 0:
                    #     Utils.showBatchImages(batch)
                    batch = batch.to(device)
                    labels = labels.to(device)
                    # model.reset_mem(batchSize, device)
                    # model.mem1 = final_membranes['mem1'].to(device)
                    # model.mem2 = final_membranes['mem2'].to(device)
                    # model.mem3 = final_membranes['mem3'].to(device)
                    # model.mem4 = final_membranes['mem4'].to(device)
                    # model.mem5 = final_membranes['mem5'].to(device)
                    startIndex = i*config.batch_size
                    endIndex = startIndex + config.batch_size

                    s, f, p, v = model(batch, config.num_time_steps_extract)
                    s = s.cpu().detach().numpy()
                    f = f.cpu().detach().numpy()
                    p = p.cpu().detach().numpy()
                    v = v.cpu().detach().numpy()
                    # print(f"v.shape: {v.shape}")
                    # s = s.cpu().detach().numpy()
                    # s = s.sum(axis=0)
                    # print(f'feats.shape is {f.shape} and probs.shape is {p.shape}')
                    spikes = s.sum(axis=0)
                    features = f.sum(axis=0)
                    probs = p.max(axis=0)
                    # print(f"v.shape: {v.shape}")
                    volts = v.mean(axis=0)
                    # print(f"volts.shape: {volts.shape}")

                    if hasattr(model, 'expansion') and model.expansion > 1:
                        # print(f"Population coding with expansion {model.expansion}")
                        spikes = spikes.reshape(spikes.shape[0], config.num_classes, model.expansion).sum(axis=2)  # or .max(axis=2)
                        probs = probs.reshape(probs.shape[0], config.num_classes, model.expansion).max(axis=2)    # Max is better for probs
                        # volts = volts.reshape(volts.shape[0], config.num_classes, model.expansion).max(axis=2)  # Max is better for volts

                    probs = Metrics.softmax(probs)
                    # if i == 0:
                    #     print(f'{i}\n{features}\n{probs}')
                        # plotProb(p, batchSize, numberOfSteps)
                    # probs = Metrics.softmax(probs)
                    # print(f'feats.shape is {features.shape} and probs.shape is {probs.shape}')
                    labels = labels.cpu().detach().numpy()
                    Spik_train[startIndex:endIndex, :] = spikes
                    Feat_train[startIndex:endIndex, :] = features
                    Prob_train[startIndex:endIndex, :] = probs
                    Volt_train[startIndex:endIndex, :] = volts
                    Tags_train.append(labels.flatten())
                    del batch, labels, features, probs
                    i += 1
                    # if i == 1: 
                    #     break
                    print(f"\rProgress: {i}", end='', flush=True)

            # Tags_train = np.array(Tags_train)
            Tags_train = np.concatenate(Tags_train)
            print()
            # print("Tags_train shape is ", Tags_train.shape)

            Spik_test = np.zeros((testDataSize, config.num_classes))
            Feat_test = np.zeros((testDataSize, featSize))
            Prob_test = np.zeros((testDataSize, config.num_classes))
            Volt_test = np.zeros((testDataSize, featSize))
            Tags_test = []
            i = 0
            with torch.no_grad():
                for batch, labels in test_loader:
                    # if i == 0:
                    #     Utils.showBatchImages(batch)
                    batch = batch.to(device)
                    labels = labels.to(device)
                    startIndex = i*config.batch_size
                    endIndex = startIndex + config.batch_size

                    s, f, p, v = model(batch, config.num_time_steps_extract)
                    s = s.cpu().detach().numpy()
                    f = f.cpu().detach().numpy()
                    p = p.cpu().detach().numpy()
                    v = v.cpu().detach().numpy()

                    spikes = s.sum(axis=0)    
                    features = f.sum(axis=0)
                    probs = p.max(axis=0)
                    volts = v.max(axis=0)
                    if hasattr(model, 'expansion') and model.expansion > 1:
                        spikes = spikes.reshape(spikes.shape[0], config.num_classes, model.expansion).sum(axis=2)  # or .max(axis=2)
                        probs = probs.reshape(probs.shape[0], config.num_classes, model.expansion).max(axis=2)    # Max is better for probs
                        # volts = volts.reshape(volts.shape[0], config.num_classes, model.expansion).max(axis=2)  # Max is better for volts

                    probs = Metrics.softmax(probs)
                    # if i == 0:
                    #     print(f'{i}\n{features}\n{probs}')
                    labels = labels.cpu().detach().numpy()
                    Spik_test[startIndex:endIndex, :] = spikes
                    Feat_test[startIndex:endIndex, :] = features
                    Prob_test[startIndex:endIndex, :] = probs
                    Volt_test[startIndex:endIndex, :] = volts
                    # Tags_test.append(labels)
                    Tags_test.append(labels.flatten())
                    del batch, labels, features, probs
                    i += 1
                    # if i == 1: 
                    #     break
                    print(f"\rProgress: {i}", end='', flush=True)

            # Tags_test = np.array(Tags_test)
            Tags_test = np.concatenate(Tags_test)
            # print("Tags_test shape is ", Tags_test.shape)
            # fileName = 'features/spike/case_' + case + '/' + dataSet_feat + '-on_' + dataSet_ID.lower() + '.npz'
            # print(fileName)
            print()
            np.savez(fileName, arr0=Spik_train, 
                            arr1=Feat_train, 
                            arr2=Prob_train, 
                            arr3=Tags_train, 
                            arr7=Spik_test, 
                            arr4=Feat_test, 
                            arr5=Prob_test, 
                            arr6=Tags_test,
                            arr8=Volt_train,
                            arr9=Volt_test)

            del Spik_train, Feat_train, Prob_train, Tags_train, Spik_test, Feat_test, Prob_test, Tags_test, Volt_train, Volt_test

        else:
            # pass
            print("Features " + str(fileName) + " already exists.")
        
    return None

def plotProb(prob, batchSize, numOfSteps):
    '''
    Plot one probability 
    '''
    print(f'Probs shape is {prob.shape}')
    # for i in range(batchSize):
    #     p = prob[:, i, 0].squeeze()

    #     plt.figure(figsize=(10, 5))
    #     plt.plot(p, marker='o')  # Convert to NumPy for plotting
    #     plt.title('Memebrane voltage')
    #     plt.xlabel('Steps')
    #     plt.ylabel('Voltage')
    #     plt.grid(True)
    #     plt.show()
    # Determine the number of columns and rows
    num_cols = 3
    num_rows = int(np.ceil(batchSize / num_cols))

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
    
    # Flatten axes for easy indexing
    axes = axes.flatten()

    for i in range(batchSize):
        p = prob[:, i, 1].squeeze()

        axes[i].plot(p, marker='o')  # Plot on the ith subplot
        axes[i].set_title(f'Membrane Voltage for Sample {i+1}')
        axes[i].set_xlabel('Steps')
        axes[i].set_ylabel('Voltage')
        axes[i].grid(True)

    # Hide any unused subplots
    for j in range(batchSize, num_rows * num_cols):
        fig.delaxes(axes[j])
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    probs_ = prob.max(axis=0)
    print(f'Max value is {probs_}')
