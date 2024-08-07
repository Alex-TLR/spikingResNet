from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10
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
        if database_name == 'MNIST':
            name = 'mnist' 
            Name = 'MNIST'
        elif database_name == 'KMNIST':
            name = 'kmnist' 
            Name = 'KMNIST'
        elif database_name == 'FMNIST':
            name = 'fmnist'
            Name = 'FashionMNIST'
        else:
            print("Wrong database name!")
            return -1

        transformData = transforms.Compose([transforms.Resize((28, 28)),
                                            transforms.Grayscale(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0,), (1,))])
    
        command_train = Name + '(root = \'data/' + name + '/\', download=True, train = True, transform = transformData)'
        print(command_train)
        command_test = Name + '(root = \'data/' + name + '/\', download=True, train = False, transform = transformData)'
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
    
