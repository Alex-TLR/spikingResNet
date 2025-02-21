from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10, SVHN, Places365, EMNIST, Food101
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os
from sklearn.manifold import TSNE

'''
Utility functions
'''

class Utils():

    def __init__(self, name):
        pass

    @staticmethod
    def load_data(database_name):
        '''
        database_name:      Name of the database (MNIST, KMNIST, FMNIST)
        '''

        transformData_gray_28 = transforms.Compose([transforms.Resize((28, 28)),
                                                    transforms.Grayscale(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0,), (1,))])
        
        transformData_rgb_32  = transforms.Compose([transforms.Resize((32, 32)),
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
            transformData = transformData_rgb_32
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

        if (database_name != 'SVHN') and (database_name != 'Places365') and (database_name != 'EMNIST') and (database_name != 'Letters') and (database_name != 'Food101'):
            command_train = Name + '(root = \'data/' + name + '/\', download=True, train = True, transform = transformData)'
            print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, train = False, transform = transformData)'
            print(command_test)
        elif database_name == 'SVHN':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'train\', target_transform = transformData)'
            print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'test\', target_transform = transformData)'
            print(command_test)
        elif database_name == 'Places365':
            command_train = Name + '(root = \'data/' + name + '/\', download=False, split = \'train-standard\', small = True, target_transform = transformData)'
            print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=False, split = \'val\', small = True, target_transform = transformData)'
            print(command_test)
        elif database_name == 'EMNIST':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'digits\', train = True, transform = transformData)'
            print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'digits\', train = False, transform = transformData)'
            print(command_test)
        elif database_name == 'Letters':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'letters\', train = True, transform = transformData)'
            print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'letters\', train = False, transform = transformData)'
            print(command_test)
        elif database_name == 'Food101':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'train\', transform = transformData)'
            print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'test\', transform = transformData)'
            print(command_test)
        else:
            print(f"Wrong data srt name.")
        
        print(command_train)
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
            print(dataset[0])
            train_tensor, _ = dataset[0]
            imageSize = train_tensor.shape
        else:
            train_tensor, _ = dataset[0]
            imageSize = train_tensor.size()
        print(f'Image size: {imageSize[0]}, {imageSize[1]}, {imageSize[2]}')

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
    def data_loader(dataset_train, dataset_test, batchSize, dataset_name, fullTrain=False):
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
            train_loader = DataLoader(train_data, batchSize, shuffle=True)
            val_loader = DataLoader(val_data, batchSize)
            test_loader = DataLoader(test_data, batchSize)

            return train_loader, val_loader, test_loader

        else:
            train_data = dataset_train
            test_data = dataset_test
            print("Train data length ", len(train_data))
            print("Test data length ", len(test_data))
            if dataset_name == 'SVHN':
                train_data = SVHNDataset(data=dataset_train.data, labels=dataset_train.labels)
                test_data = SVHNDataset(data=dataset_test.data, labels=dataset_test.labels)
                train_loader = DataLoader(train_data, batchSize, shuffle=True)
                test_loader = DataLoader(test_data, batchSize, shuffle=False)
            else:
                train_loader = DataLoader(train_data, batchSize, shuffle=True)
                test_loader = DataLoader(test_data, batchSize, shuffle=False)

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
    def find_threshold(test_Y, test_dist, positive_label, drop = True): 
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

        fpr, tpr, thresholds = roc_curve(test_Y, test_dist, pos_label = positive_label, drop_intermediate = drop)
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

        print(f"Utils, th_tpr80: {th_tpr80}, th_tpr95: {th_tpr95}")

        return th_tpr80, th_tpr95


    @staticmethod
    def plot_train_val_stats(history):
        '''
        Plot history
        '''
        pass 

    @staticmethod
    def visualize_feature(Id, Ood, case):
        '''
            Plot reduced features Id/Ood
        
            Id (str):   In distribution dataset
            Ood (str):  Out of distribution dataset  

            feat_train = F['arr1']  
            prob_train = F['arr2']  
            tags_train = F['arr3']  
            feat_test  = F['arr4']  
            prob_test  = F['arr5']  
            tags_test  = F['arr6']  
        
        '''

        fileName = 'tsne/' + str(case) + '_' + str(Id) + '_' + str(Ood) + '.npz'
        print(fileName)
        
        if (Utils.does_file_exists(fileName)):
            
            tsne = TSNE(n_components=2, random_state=42)
            folderName = 'features/spike/case_' + case + '/'
            print(folderName)
            files = os.listdir(folderName)
            print(files)

            filePathId = folderName + str(Id) + '-on_' + Id.lower() + '.npz'
            filePathOod = folderName + str(Ood) + '-on_' + Id.lower() + '.npz'

            features = []
            border = 0

            F_id = np.load(filePathId)
            features1 = F_id['arr4']
            border = len(features1)

            F_ood = np.load(filePathOod)
            features2 = F_ood['arr1']

            print(f'features1.shape is {features1.shape}')
            print(f'features2.shape is {features2.shape}')
            features = np.vstack((features1, features2))
            print(f'features.shape is {features.shape}')
            features_tsne = tsne.fit_transform(features)

            features1_tsne = features_tsne[:border]
            features2_tsne = features_tsne[border:]

            np.savez(fileName, arr1=features1_tsne, arr2=features2_tsne)
            print(f'features_tsne shape is {features_tsne.shape}')

        else:
            # Load data
            F = np.load(fileName)
            features1_tsne = F['arr1']
            features2_tsne = F['arr2']

        # available_fonts = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
        # print(available_fonts)

        plt.figure(figsize=(10, 8))

        # Plot the first set of features
        plt.scatter(features1_tsne[:, 0], features1_tsne[:, 1], marker='.', c='blue', label=str(Id), alpha=0.5)

        # Plot the second set of features
        plt.scatter(features2_tsne[:, 0], features2_tsne[:, 1], marker='x', c='orange', label=str(Ood), alpha=0.5)

        # Add labels and legend
        # plt.xlabel('t-SNE Dimension 1')
        # plt.ylabel('t-SNE Dimension 2'

        plt.title('t-SNE Visualization of Feature Vectors', fontname='Liberation Serif', fontsize=18)
        plt.legend(fontsize=18)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        ax = plt.gca()  # Get current axis
        # for spine in ax.spines.values():
        #     spine.set_visible(False)
        # plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Show the plot
        plt.show()

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
    
