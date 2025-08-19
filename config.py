    # Case 01:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: spikeConvNN1 (ResNetModel = 1)
    # Number of classes: 10
    # Batch size: 32 

    # Case 02:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet9Model (ResNetModel = 9)
    # Number of classes: 10
    # Batch size: 32

    # Case 03:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet10Model (ResNetModel = 10)
    # Number of classes: 10
    # Batch size: 32

    # Case 04:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: spikeConvNN2 (ResNetModel = 2)
    # Number of classes: 10
    # Batch size: 32

    # Case 06:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet18Model (ResNetModel = 18)
    # Number of classes: 10
    # Batch size: 32
    # Expansion must be set to 1 in order to use regular model 

    # Case 07:
    # spike-ResNet10 model with trainable initialization
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet10Model (ResNetModel = 10)
    # Number of classes: 10
    # Batch size: 32

    # Case 08:
    # SEW model ResNet18 test accuracy is 82.53%
    # Network model: SEW ResNet18 (ResNetModel = 22)

    # Case 09:
    # spike-ResNet20 model 
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet20Model (ResNetModel = 20)
    # Number of classes: 10
    # Batch size: 32

    # Case 10:
    # Network model: SpikeResNet20Model (ResNetModel = 20)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 4
    # Expansion: E = 1
    # Auto augmentation: A = False

    ##########################################################

    # Case 11:
    # Network model: SpikeResNet10Model (ResNetModel = 10)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 4
    # Expansion: E = 1
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer

    # Case 12:
    # Network model: SpikeResNet10Model (ResNetModel = 10)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 1
    # Expansion: E = 50
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer

    # Case 13:
    # Network model: SpikeResNet10Model (ResNetModel = 10)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 1
    # Expansion: E = 50
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer
    # Population coding, but for feature extraction more time steps are used
    # Different from the previous case. More time steps are used for feature extraction.

    # Case 14:
    # Network model: SpikeResNet18Model (ResNetModel = 18)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 4
    # Expansion: E = 1
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer

    # Case 15:
    # Network model: SpikeResNet18Model (ResNetModel = 18)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 1
    # Expansion: E = 50
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer
    # Population coding, but for feature extraction more time steps are used

    # Case 16:
    # Network model: SpikeResNet4Model (ResNetModel = 4)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 4
    # Expansion: E = 1
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer

    # Case 17:
    # Network model: SpikeResNet4Model (ResNetModel = 4)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 1
    # Expansion: E = 50
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer

    # Case 18:
    # Network model: SpikeResNet4Model (ResNetModel = 4)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 1
    # Expansion: E = 50
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer
    # Population coding, but for feature extraction more time steps are used

    # Case 19:
    # Network model: SpikeResNet4Model (ResNetModel = 4)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 50
    # Expansion: E = 1
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer


class ExperimentConfig:
    def __init__(self):
        # Dataset parameters
        self.dataset_ID = 'CIFAR10'
        self.dataset_feat = ['CIFAR10', 'SVHN', 'Food101']
        # self.dataset_ID = 'KMNIST'
        # self.dataset_feat = ['MNIST', 'FMNIST', 'KMNIST', 'Letters']
        self.case = '04'
        
        # Model parameters
        self.model_type = 'spike'
        self.resnet_model = 2
        self.num_classes = 10
        self.expansion = 1
        self.num_time_steps_train = 50
        self.num_time_steps_extract = 50
        
        # Training parameters
        self.batch_size = 64
        self.epochs = 200
        self.full_train = True
        self.auto_aug = False
        self.pretrained = False

        # Parser defaults (only used if explicitly provided)
        self.seed = 42
        self.device = 'cuda'
        self.mode = 'test'
        self.num_workers = 4
        self.learning_rate = 0.001
        self.optimizer = 'adam'
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.scheduler = 'cosine'
        self.gradient_clipping = 0.5
        self.step_size = 30
    
    def update_from_args(self, args, provided_args):
        """Update config only with explicitly provided command line arguments"""
        # Only override if argument was explicitly provided (not default)
        if 'dataset_ID' in provided_args:
            self.dataset_ID = args.dataset_ID
        if 'case' in provided_args:
            self.case = args.case
        if 'model' in provided_args:
            self.resnet_model = args.model
        if 'num_classes' in provided_args:
            self.num_classes = args.num_classes
        if 'batch_size' in provided_args:
            self.batch_size = args.batch_size
        if 'epochs' in provided_args:
            self.epochs = args.epochs
        if 'pretrained' in provided_args:
            self.pretrained = args.pretrained
        if 'seed' in provided_args:
            self.seed = args.seed
        if 'device' in provided_args:
            self.device = args.device
        if 'mode' in provided_args:
            self.mode = args.mode
        if 'num_workers' in provided_args:
            self.num_workers = args.num_workers
        if 'learning_rate' in provided_args:
            self.learning_rate = args.learning_rate
        if 'optimizer' in provided_args:
            self.optimizer = args.optimizer
        if 'momentum' in provided_args:
            self.momentum = args.momentum
        if 'weight_decay' in provided_args:
            self.weight_decay = args.weight_decay
        if 'scheduler' in provided_args:
            self.scheduler = args.scheduler
        if 'step_size' in provided_args:
            self.step_size = args.step_size
    
    # def to_dict(self):
    #     """Convert config to dictionary for function calls"""
    #     return {
    #         # Core experiment parameters
    #         'dataSet_ID': self.dataset_ID,
    #         'modelType': self.model_type,
    #         'batchSize': self.batch_size,
    #         'numberOfClasses': self.num_classes,
    #         'ResNetModel': self.resnet_model,
    #         'expansion': self.expansion,
    #         'epochs': self.epochs,
    #         'fullTrain': self.full_train,
    #         'auto_aug': self.auto_aug,
    #         'pretrained': self.pretrained,
    #         'case': self.case,
    #         'dataSet_feat': self.dataset_feat,
            
    #         # Parser parameters (available when needed)
    #         'seed': self.seed,
    #         'device': self.device,
    #         'mode': self.mode,
    #         'num_workers': self.num_workers,
    #         'learning_rate': self.learning_rate,
    #         'optimizer': self.optimizer,
    #         'momentum': self.momentum,
    #         'weight_decay': self.weight_decay,
    #         'scheduler': self.scheduler,
    #         'step_size': self.step_size
    #     }
