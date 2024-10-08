from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10, SVHN, Places365, EMNIST
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os

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
        else:
            print("Wrong database name!")
            return -1

        if (database_name != 'SVHN') and (database_name != 'Places365') and (database_name != 'EMNIST') and (database_name != 'Letters'):
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
            # print("jesmo li ovdje")
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'letters\', train = True, transform = transformData)'
            print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'letters\', train = False, transform = transformData)'
            print(command_test)
        
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
                test_loader = DataLoader(test_data, batchSize)
            else:
                train_loader = DataLoader(train_data, batchSize, shuffle=True)
                test_loader = DataLoader(test_data, batchSize)

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

        _, tpr, thresholds = roc_curve(test_Y, test_dist, pos_label = positive_label, drop_intermediate = drop)

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

        return th_tpr80, th_tpr95


    @staticmethod
    def plot_train_val_stats(history):
        '''
        Plot history
        '''
        pass 


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
    
