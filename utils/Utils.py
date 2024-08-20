from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10, SVHN
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch

import matplotlib.pyplot as plt

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
        elif database_name == 'CIFAR10':
            name = 'cifar10'
            Name =  'CIFAR10'
            transformData = transformData_rgb_32
        elif database_name == 'SVHN':
            name = 'svhn'
            Name =  'SVHN'
            transformData = transformData_rgb_32
        else:
            print("Wrong database name!")
            return -1

        if database_name != 'SVHN':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, train = True, transform = transformData)'
            print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, train = False, transform = transformData)'
            print(command_test)
        elif database_name == 'SVHN':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'train\', target_transform = transformData)'
            print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'test\', target_transform = transformData)'
            print(command_test)
        
        dataset_train = eval(command_train)
        dataset_test = eval(command_test)

        return dataset_train, dataset_test
    

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
    def plot_train_val_stats(history):
        '''
        Plot history
        '''
        pass 
    
