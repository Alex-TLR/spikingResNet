from cProfile import label
from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10, SVHN, Places365, EMNIST, Food101
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import os
from sklearn.manifold import TSNE

'''
Utility functions
'''

class Utils():

    def __init__(self, name):
        pass

    def targetTransform(label):
        return label - 1 

    @staticmethod
    def load_data(database_name, auto_aug=False):
        '''
        database_name:      Name of the database (MNIST, KMNIST, FMNIST)
        '''

        transformData_gray_28 = transforms.Compose([transforms.Resize((28, 28)),
                                                    # transforms.RandomCrop(28, padding=4),
                                                    # transforms.RandomHorizontalFlip(),
                                                    # transforms.Grayscale(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0,), (1,))
                                                    # transforms.Normalize((0.1918,), (0.3483,))
                                                    ])
        
        # KMNIST transforms.Normalize((0.1918,), (0.3483,))

        transformData_rgb_32  = transforms.Compose([transforms.Resize((32, 32)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0,0,0,), (1,1,1,))])
        
        # Updated CIFAR10 transform with AutoAugment support
        def get_cifar10_transforms(auto_aug=True, cutout=False, training=True):
            if training:
                aug = []
                aug.append(transforms.Resize((32, 32)))
                aug.append(transforms.RandomCrop(32, padding=4))
                aug.append(transforms.RandomHorizontalFlip())
                
                if auto_aug:
                    aug.append(CIFAR10Policy())  # Add AutoAugment policy
                
                aug.append(transforms.ToTensor())
                
                if cutout:
                    # You'll need to implement Cutout class or import it
                    # aug.append(Cutout(n_holes=1, length=16))
                    pass
                
                # aug.append(transforms.Normalize(
                #     (0.4914, 0.4822, 0.4465), 
                #     (0.2023, 0.1994, 0.2010)
                # ))
                aug.append(transforms.Normalize((0,0,0,), (1,1,1,)))
                
                return transforms.Compose(aug)
            else:
                # Test/validation transform (no augmentation)
                # return transforms.Compose([
                #     transforms.Resize((32, 32)),
                #     transforms.ToTensor(),
                #     transforms.Normalize(
                #         (0.4914, 0.4822, 0.4465), 
                #         (0.2023, 0.1994, 0.2010)
                #     )
                # ])
                return transforms.Compose([transforms.Resize((32, 32)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0,0,0,), (1,1,1,))])
        
        if database_name == 'MNIST':
            name = 'mnist' 
            Name = 'MNIST'
            transformData = transformData_gray_28
        elif database_name == 'KMNIST':
            name = 'kmnist' 
            Name = 'KMNIST'
            transformData = transformData_gray_28
        elif database_name == 'FMNIST':
            name = 'fmnist'
            Name = 'FashionMNIST'
            transformData = transformData_gray_28
        elif database_name == 'EMNIST':
            name = 'emnist'
            Name = 'EMNIST'
            transformData = transformData_gray_28
        elif database_name == 'Letters':
            name = 'letters'
            Name = 'EMNIST'
            transformData = transformData_gray_28
        elif database_name == 'CIFAR10':
            name = 'cifar10'
            Name =  'CIFAR10'
            # transformData = transformData_cifar10
            transformData_train = get_cifar10_transforms(auto_aug=auto_aug, training=True)
            transformData_test = get_cifar10_transforms(auto_aug=False, training=False)
            transformData_train = transformData_rgb_32
            transformData_test = transformData_rgb_32
        elif database_name == 'SVHN':
            name = 'svhn'
            Name =  'SVHN'
            transformData = transformData_rgb_32
        elif database_name == 'Places365':
            name = 'places'
            Name =  'Places365'
            transformData = transformData_rgb_32
        elif database_name == 'Food101':
            name = 'food'
            Name =  'Food101'
            transformData = transformData_rgb_32
        else:
            print("Wrong database name!")
            return -1

        # if (database_name != 'SVHN') and (database_name != 'Places365') and (database_name != 'EMNIST') and (database_name != 'Letters') and (database_name != 'Food101'):
        #     command_train = Name + '(root = \'data/' + name + '/\', download=True, train = True, transform = transformData)'
        #     # print(command_train)
        #     command_test = Name + '(root = \'data/' + name + '/\', download=True, train = False, transform = transformData)'
        #     # print(command_test)
        if database_name == 'CIFAR10':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, train = True, transform = transformData_train)'
            command_test = Name + '(root = \'data/' + name + '/\', download=True, train = False, transform = transformData_test)'
        elif database_name == 'SVHN':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'train\', target_transform = transformData)'
            # print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'test\', target_transform = transformData)'
            # print(command_test)
        elif database_name == 'Places365':
            command_train = Name + '(root = \'data/' + name + '/\', download=False, split = \'train-standard\', small = True, target_transform = transformData)'
            # print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=False, split = \'val\', small = True, target_transform = transformData)'
            # print(command_test)
        elif database_name == 'EMNIST':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'digits\', train = True, transform = transformData)'
            # print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'digits\', train = False, transform = transformData)'
            # print(command_test)
        elif database_name == 'Letters':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'letters\', train = True, transform = transformData, target_transform = Utils.targetTransform)'
            print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'letters\', train = False, transform = transformData, target_transform = Utils.targetTransform)'
            print(command_test)
        elif database_name == 'Food101':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'train\', transform = transformData)'
            # print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'test\', transform = transformData)'
            # print(command_test)
        elif database_name == "MNIST":
            command_train = Name + '(root = \'data/' + name + '/\', download=True, train = True, transform = transformData)'
            command_test = Name + '(root = \'data/' + name + '/\', download=True, train = False, transform = transformData)'
        elif database_name == "FMNIST":
            command_train = Name + '(root = \'data/' + name + '/\', download=True, train = True, transform = transformData)'
            command_test = Name + '(root = \'data/' + name + '/\', download=True, train = False, transform = transformData)'
        elif database_name == "KMNIST":
            command_train = Name + '(root = \'data/' + name + '/\', download=True, train = True, transform = transformData)'
            command_test = Name + '(root = \'data/' + name + '/\', download=True, train = False, transform = transformData)'
        else:
            print(f"Wrong data set name.")

        # print(command_train)
        dataset_train = eval(command_train)
        dataset_test = eval(command_test)

        return dataset_train, dataset_test
    

    @staticmethod
    def get_image_size(dataset, name):
        '''
        dataset:        loaded dataset
        name:           database name 
        '''

        if (name == 'SVHN'):
            train_tensor = dataset.data[0]
            imageSize = train_tensor.shape
        elif (name == 'Food101'):
            # print(dataset[0])
            train_tensor, _ = dataset[0]
            imageSize = train_tensor.shape
        else:
            train_tensor, _ = dataset[0]
            imageSize = train_tensor.size()
        # print(f'Image size: {imageSize[0]}, {imageSize[1]}, {imageSize[2]}')

        return imageSize[0], imageSize[1], imageSize[2]
    

    @staticmethod
    def make_features_dir(modelType, case = '00'):

        folderPath = 'features/' + modelType + '/case_' + case
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
            print(f"Directory {folderPath} created.")
        else:
            print(f"Directory {folderPath} already exists.")

        return None
    
    @staticmethod
    def does_file_exists(fileName):

        '''
        Reverse logic
        '''

        if os.path.exists(fileName):
            print(f"The file {fileName} exists.")
            return False
        else:
            print(f"The file {fileName} does not exist.")
            return True

    @staticmethod
    def data_loader(dataset_train, dataset_test, batchSize, dataset_name, fullTrain=False, worker_init_fn=None, generator=None):
        '''
        Define data loaders
        '''
        if (fullTrain == False):
            print("Define training/validation split ...")
            dataSize = len(dataset_train)
            tSize = int(0.8 * dataSize)
            vSize = dataSize - tSize

            train_data, val_data = random_split(dataset_train, [tSize, vSize])
            test_data = dataset_test
            if dataset_name == 'SVHN':
                train_data = SVHNDataset(data=train_data.dataset.data, labels=train_data.dataset.labels)
                val_data = SVHNDataset(data=val_data.dataset.data, labels=val_data.dataset.labels)
                test_data = SVHNDataset(data=test_data.data, labels=test_data.labels)

            print("Train data length ", len(train_data))
            print("Valid data length ", len(val_data))
            print("Test data length ", len(test_data))
            # train_loader = DataLoader(train_data, batchSize, shuffle=True)
            # val_loader = DataLoader(val_data, batchSize)
            # test_loader = DataLoader(test_data, batchSize)

            # Add seeding parameters to DataLoaders
            train_loader = DataLoader(
                train_data, 
                batchSize, 
                shuffle=True,
                worker_init_fn=worker_init_fn,
                generator=generator
            )
            val_loader = DataLoader(
                val_data, 
                batchSize,
                worker_init_fn=worker_init_fn,
                generator=generator
            )
            test_loader = DataLoader(
                test_data, 
                batchSize,
                worker_init_fn=worker_init_fn,
                generator=generator
            )

            return train_loader, val_loader, test_loader

        else:
            train_data = dataset_train
            test_data = dataset_test
            # print("Train data length ", len(train_data))
            # print("Test data length ", len(test_data))
            if dataset_name == 'SVHN':
                train_data = SVHNDataset(data=dataset_train.data, labels=dataset_train.labels)
                test_data = SVHNDataset(data=dataset_test.data, labels=dataset_test.labels)
                # train_loader = DataLoader(train_data, batchSize, shuffle=True)
                # test_loader = DataLoader(test_data, batchSize, shuffle=False)
                train_loader = DataLoader(
                    train_data, 
                    batchSize, 
                    shuffle=True,
                    worker_init_fn=worker_init_fn,
                    generator=generator
                )
                test_loader = DataLoader(
                    test_data, 
                    batchSize, 
                    shuffle=False,
                    worker_init_fn=worker_init_fn,
                    generator=generator
                )
            else:
                # train_loader = DataLoader(train_data, batchSize, shuffle=True)
                # test_loader = DataLoader(test_data, batchSize, shuffle=False)
                train_loader = DataLoader(
                    train_data, 
                    batchSize, 
                    shuffle=True,
                    worker_init_fn=worker_init_fn,
                    generator=generator
                )
                test_loader = DataLoader(
                    test_data, 
                    batchSize, 
                    shuffle=False,
                    worker_init_fn=worker_init_fn,
                    generator=generator
                )

            return train_loader, test_loader

  
    @staticmethod
    def showBatch(inputData):
        '''
        Show the batch of images in the grid
        '''
        for images, labels in inputData:
            fig, ax = plt.subplots(figsize=(12,6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            plt.show()
            break

    @staticmethod
    def showBatchImages(inputImages):
        '''
        Show the batch of images in the grid
        '''
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.imshow(make_grid(inputImages, nrow=8).permute(1, 2, 0))  # Arrange images in a grid
        plt.show()


    @staticmethod
    def get_device():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        return device 
    

    @staticmethod
    def find_threshold(test_labels, test_dist, positive_label, drop = True): 
        '''
        Find threshold values input values

        Inputs:
            test_Y:         an array with train labels
            test_dist:      an array of test features distances 
            positive_label: set if 1 is positive label, or 0 is positive label

        Outputs:
            th_tpr80:       threshold for which TPR is 80 percent
            th_tpr95:       threshold for which TPR is 95 percent
        '''

        test_dist = np.array(test_dist)
        fpr, tpr, thresholds = roc_curve(test_labels, test_dist, pos_label = positive_label, drop_intermediate = drop)
        # print(f"thresholds: {thresholds.min()}, {thresholds.max()}")
        # Plot ROC curve
        # plt.figure()
        # plt.plot(fpr, tpr, marker='o', linestyle='-', color='b')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve')
        # plt.grid(True)
        # plt.show()

        dtpr80 = np.absolute(tpr-0.8)
        th_tpr80 = thresholds[dtpr80.argmin()]

        dtpr95 = np.absolute(tpr-0.95)
        th_tpr95 = thresholds[dtpr95.argmin()]

        # predictions = np.zeros(len(test_dist))
        # for i in range(len(test_dist)):    
        #     predictions[i] = 0 if test_dist[i] < th_tpr95 else 1

        # print(f"Result: {np.sum(a == b for a, b in zip(test_labels, predictions) if a == 1)  / len(predictions[test_labels == 1])}")
        # print(f"Utils, th_tpr80: {th_tpr80}, th_tpr95: {th_tpr95}")

        return th_tpr80, th_tpr95

    @staticmethod
    def create_population_targets(labels, num_classes=10, pop_per_class=50):
        batch_size = labels.size(0)
        targets = torch.zeros(batch_size, num_classes * pop_per_class, device=labels.device)
        
        for i, label in enumerate(labels):
            start_idx = label * pop_per_class
            end_idx = start_idx + pop_per_class
            targets[i, start_idx:end_idx] = 1.0
        
        return targets


    @staticmethod
    def plot_train_val_stats(history):
        '''
        Plot history
        '''
        pass 

    @staticmethod
    def visualize_feature(Id, Ood, case, feature_type):
        '''
            Plot reduced features Id/Ood
        
            Id (str):   In distribution dataset
            Ood (str):  Out of distribution dataset  

            spik_train = F['arr0']
            feat_train = F['arr1']  
            prob_train = F['arr2']  
            tags_train = F['arr3']
            volt_train = F['arr8']
            spik_test  = F['arr7']  
            feat_test  = F['arr4']  
            prob_test  = F['arr5']  
            tags_test  = F['arr6']
            volt_test  = F['arr9']  
        
        '''

        fileName = 'tsne/' + str(case) + '_' + str(Id) + '_' + str(Ood) + '_feature_type_' + str(feature_type) + '.npz'
        print(fileName)
        
        if (Utils.does_file_exists(fileName)):
            
            tsne = TSNE(n_components=2, random_state=42)
            folderName = 'features/spike/case_' + case + '/'
            print(folderName)
            files = os.listdir(folderName)
            print(files)

            filePathId = folderName + str(Id) + '-on_' + Id.lower() + '.npz'
            if Ood is not None:
                filePathOod = folderName + str(Ood) + '-on_' + Id.lower() + '.npz'

            features = []
            border = 0

            F_id = np.load(filePathId)
            if feature_type == 'features':
                features1 = F_id['arr4']
            elif feature_type == 'spikes':
                features1 = F_id['arr0']
            elif feature_type == 'probs':
                features1 = F_id['arr2']
            elif feature_type == 'voltages':
                features1 = F_id['arr8']

            if Ood is not None:
                border = len(features1)
                F_ood = np.load(filePathOod)
                if feature_type == 'features':
                    features2 = F_ood['arr4']
                elif feature_type == 'spikes':
                    features2 = F_ood['arr0']
                elif feature_type == 'probs':
                    features2 = F_ood['arr2']
                elif feature_type == 'voltages':
                    features2 = F_ood['arr8']

            print(f'features1.shape is {features1.shape}')

            if Ood is not None:
                print(f'features2.shape is {features2.shape}')
                features = np.vstack((features1, features2))
            else:
                features = features1 
            print(f'features.shape is {features.shape}')
            features_tsne = tsne.fit_transform(features)

            if Ood is not None:
                features1_tsne = features_tsne[:border]
                features2_tsne = features_tsne[border:]

            if Ood is not None:
                np.savez(fileName, arr1=features1_tsne, arr2=features2_tsne)
            else:
                np.savez(fileName, arr1=features_tsne)
            print(f'features_tsne shape is {features_tsne.shape}')

        else:
            # Load data
            F = np.load(fileName)
            features1_tsne = F['arr1']
            features2_tsne = F['arr2']

        # available_fonts = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
        # print(available_fonts)

        plt.rcParams.update({
            'text.usetex': True,  # Use LaTeX for rendering text
            'font.family': 'serif',  # Set the base font to serif
            'font.serif': ['Times New Roman'],  # Set Times New Roman (if available)
            'mathtext.fontset': 'stix',  # Use the STIX math font family (similar to IEEE)
            'axes.titlesize': 24,
            'axes.labelsize': 24,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'figure.titlesize': 24,
        })

        plt.figure(figsize=(10, 8))

        if Ood is not None:
            # Plot the first set of features
            plt.scatter(features1_tsne[:, 0], features1_tsne[:, 1], marker='.', c='blue', label=str(Id), alpha=0.5)
            # Plot the second set of features
            plt.scatter(features2_tsne[:, 0], features2_tsne[:, 1], marker='x', c='orange', label=str(Ood), alpha=0.5)
        else:
            plt.scatter(features_tsne[:, 0], features_tsne[:, 1], marker='.', c='blue', label=str(Id), alpha=0.5)

        # Add labels and legend
        # plt.xlabel('t-SNE Dimension 1')
        # plt.ylabel('t-SNE Dimension 2'
        # , fontname='Liberation Serif',

        if case == '03':
            plt.title('t-SNE visualization of spike-ResNet10 feature vectors', fontsize=24)
        elif case == '04':
            plt.title('t-SNE visualization of spike-Conv feature vectors', fontsize=24)
        elif case == '06':
            plt.title('t-SNE visualization of spike-ResNet18 feature vectors', fontsize=24)
        plt.legend(fontsize=16)
        plt.xlabel('t-SNE-1')
        plt.ylabel('t-SNE-2')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # ax = plt.gca()  # Get current axis
        # # for spine in ax.spines.values():
        # #     spine.set_visible(False)
        # # plt.tight_layout(pad=0)
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # # Show the plot
        # plt.show()

        fileName = 'tsne/' + str(case) + '_' + str(Id) + '_' + str(Ood) + '.pdf'
        plt.savefig(fileName, format="pdf", dpi=300, bbox_inches="tight", transparent=False)

        # Initialize interactive mode
        # plt.ion()

        # # Create a 3D plot
        # fig = plt.figure(figsize=(12, 10))
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot the first set of features
        # ax.scatter(features1_tsne[:, 0], features1_tsne[:, 1], features1_tsne[:, 2], 
        #         c='blue', label='Dataset 1', alpha=0.6, s=50)

        # # Plot the second set of features
        # ax.scatter(features2_tsne[:, 0], features2_tsne[:, 1], features2_tsne[:, 2], 
        #         c='red', label='Dataset 2', alpha=0.6, s=50)

        # # Add labels and legend
        # ax.set_xlabel('t-SNE Dimension 1')
        # ax.set_ylabel('t-SNE Dimension 2')
        # ax.set_zlabel('t-SNE Dimension 3')
        # ax.set_title('3D t-SNE Visualization of Feature Vectors')
        # ax.legend()

        # # Show the plot
        # plt.show()

        # # Disable interactive mode
        # plt.ioff()

        return None
    
    @staticmethod
    def prediction_spikes(spikes):
        '''
        Function that calculates the predicted class for each sample in the batch. The sample
        is a feature vector, i.e. spike pattern. Prediction is done as an argmax over the spike pattern.
        The predicted class is the one with the highest spike count.

        Inputs:
            spikes (np.array): tensor with the spike patterns (batch_size, num_classes)

        Outputs:
            predicted (np.array): array with the predicted class for each sample (batch_size,)
        '''
        return np.argmax(spikes, axis=1)
    
    @staticmethod
    def prediction_probs(probs):
        '''
        Function that calculates the predicted class for each sample in the batch. The sample
        is a vectro with class probabilities. The predicted class is the one with the highest probability.

        Inputs:
            probs (np.array): tensor with the sample class probabilities (batch_size, num_classes)

        Outputs:
            predicted (np.array): array with the predicted class for each sample (batch_size,)
        '''
        return np.argmax(probs, axis=1)
    
    @staticmethod
    def distance_from_centroids(spikes, centroids, net_outputs, prob):
        '''
        Function that computes the distances for each feature vector (spike pattern) from the each centroid
        given as input parameter.

        Inputs:
            spikes (np.array): tensor with the spike patterns (batch_size, feature_size)
            centroids (list): list of cluster centroids
            net_outputs (np.array): tensor with the network outputs (batch_size, num_classes)
            
            prob (np.array): tensor with the sample class probabilities (batch_size, num_classes)

        Outputs:
            result (list): list of distances for each class
        '''
        
        print(f"centroids.shape: {centroids.shape}")

        # Initialize the result list with empty lists for each class
        result = [[]] * len(centroids)

        # Check if the predictions based on spikes and probabilities are the same
        predictions_1 = Utils.prediction_spikes(net_outputs) 
        predictions_2 = Utils.prediction_probs(prob)

        compare_predictions = np.array_equal(predictions_1, predictions_2)
        print(f"Are predictions the same: {compare_predictions}")

        for i, spike in enumerate(spikes):
            # Get the predicted class for the current spikes
            pred = predictions_1[i]
            cent = np.array(centroids[pred])

            # Compute minimum distance for each test sample for corresponding class
            # Euclidan distance
            min_dist = -np.min(np.sqrt(np.sum((cent - spike) ** 2, axis=1)))
            # Manhattan distance
            # min_dist = np.min(np.sum(np.abs(cent - spike), axis=1))
            result[pred].append(min_dist)
    
        return [np.array(x) for x in result]
    
    @staticmethod
    def distance_from_class_centroids(spikes, centroids):
        '''
        Function that computes the distances for each feature vector (spike pattern) from 
        all class centroids.

        Used in NCM.

        Inputs:
            spikes (np.array): tensor with the spike patterns (batch_size, feature_size)
            centroids (list): list of cluster centroids

        Outputs:
            result (list): list of distances for each class
        '''

        # Compute the squared Euclidean distances between spikes and centroids
        # Using broadcasting to calculate distances in a vectorized manner
        diff = spikes[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # Shape: (batch_size, num_classes, feature_size)
        # dist_squared = np.sum(diff**2, axis=2)  # Shape: (batch_size, num_classes)

        # # Find the minimum distance for each spike and negate it
        # distances = -np.sqrt(np.min(dist_squared, axis=1))  # Shape: (batch_size,)

        distances = -np.min(np.linalg.norm(diff, axis=2), axis=1)      # (batch_size,)

        return distances

    # ### Explanation of Optimizations:
    # 1. **Broadcasting**:
    #    - Instead of iterating over each spike and centroid, we use NumPy's broadcasting to compute the pairwise differences between all spikes and centroids in one step.
    #    - `spikes[:, np.newaxis, :]` expands the `spikes` array to shape `(batch_size, 1, feature_size)`.
    #    - `centroids[np.newaxis, :, :]` expands the `centroids` array to shape `(1, num_classes, feature_size)`.
    #    - The subtraction results in a shape of `(batch_size, num_classes, feature_size)`.

    # 2. **Vectorized Squared Sum**:
    #    - The squared Euclidean distance is computed using `np.sum(diff**2, axis=2)` for all spikes and centroids simultaneously, avoiding the inner loop.

    # 3. **Efficient Minimum Calculation**:
    #    - The minimum distance for each spike is computed using `np.min(dist_squared, axis=1)`.

    # 4. **Avoid Explicit Loops**:
    #    - By replacing the nested loops with vectorized operations, the function becomes significantly faster, especially for large datasets.

    # ### Performance Improvement:
    # - The optimized function eliminates the `O(batch_size * num_classes)` complexity of the nested loops and replaces it with efficient NumPy operations that leverage low-level optimizations.
    # - This will result in a significant speedup, especially when `batch_size` or `num_classes` is large.
    
    @staticmethod
    def id_ood_predictions(distances, threshold):
        '''
        Make In-distribution / Out-of-distribution predictions based on the distances and threshold.

        Inputs:
            distances (list): list of distances for each class
            threshold (float): threshold value for classification

        Outputs:
            in_or_out_distribution (np.array): array with True if InD and False if OOD
        '''

        predictions = np.zeros(len(distances), dtype=np.int32)
        for i in range(len(distances)):    
            predictions[i] = 1 if distances[i] > threshold else 0

        return predictions


class SVHNDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)  
        self.labels = torch.tensor(labels, dtype=torch.long)  
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Apply transformation if any
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def get_preds_from_spike_vector(spikes):
    return spikes.argmax(axis=1)

def get_preds_from_probs_vector(probs):
    return probs.argmax(axis=1)

def distances_from_average_clusters(spikes, net_outputs, clusters, prob):
    result = [[] for i in range(len(clusters))]

    preds = get_preds_from_spike_vector(net_outputs) 
    pred2 = get_preds_from_probs_vector(prob) 

    for i, spike in enumerate(spikes):
        pred = preds[i]
        cent = np.array(clusters[pred])

        # Compute minimum distance for each test semple for corresponding class
        min_dist = np.min(np.sum(np.abs(cent - spike), axis=1))
        result[pred].append(min_dist)
    
    return [np.array(x) for x in result]


import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
        #     operation1, ranges[operation1][magnitude_idx1],
        #     operation2, ranges[operation2][magnitude_idx2])
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img

class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
