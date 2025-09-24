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
    # Case 12b trains on mse and E = 1, mse_loss, it is like popCode but with no expansion
    # Case 12c trains on mse and E = 5, that is population coding
    # Case 12d trains on mse and E = 10, that is population coding with more features

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
    # Case 15b trains on mse and E = 1, mse_loss, it is like popCode but with no expansion
    # Case 15c trains on mse and E = 5, that is population coding
    # Case 15d trains on mse and E = 10, that is population coding with more features

    # Case 16:
    # Network model: SpikeResNet4Model (ResNetModel = 4)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 4
    # Expansion: E = 1
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer
    # self.loss = 'cross_entropy'
    # self.fit = 'membrane'

    # Case 17:
    # Network model: SpikeResNet4Model (ResNetModel = 4)
    # Training parameters
    # Number of classes: 10
    # Batch size: 64    
    # Time steps: T = 1
    # Expansion: E = 50
    # Auto augmentation: A = False
    # Take voltages of the preultimate membrane voltage layer
    # Case 17b trains on mse and E = 1, mse_loss, it is like popCode but with no expansion
    # Case 17c trains on mse and E = 5, that is population coding
    # Case 17d trains on mse and E = 10, that is population coding

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

    # Case 20:
    # Network model: SpikeResNet10Model (ResNetModel = 10)
    # Training parameters
    # Number of classes: 10
    # Batch size: 32    
    # Time steps: T = 50
    # Expansion: E = 1
    # Auto augmentation: A = False
    # mse_loss

    # Case 21:
    # InDistribution: CIFAR10
    # OutOfDistribution: 'MNIST', 'SVHN', 'Textures', 'Places365'
    # Network model: SpikeResNet18Model (ResNetModel = 18)
    # Training parameters
    # Number of classes: 10
    # Batch size: 64    
    # Time steps: T = varying
    # Expansion: E = 1 / 50
    # Auto augmentation: A = False
    # mse_loss
    # Epochs = 200

    # Case 22:
    # InDistribution: CIFAR10
    # OutOfDistribution: 'MNIST', 'SVHN', 'Textures', 'Places365'
    # Network model: SpikeResNet10Model (ResNetModel = 10)
    # Training parameters
    # Number of classes: 10
    # Batch size: 64    
    # Time steps: T = varying
    # Expansion: E = 1 / 50
    # Auto augmentation: A = False
    # mse_loss
    # Epochs = 200

from email import parser


class ExperimentConfig:
    def __init__(self):
        # Dataset parameters
        self.dataset_ID = 'CIFAR10'
        # self.dataset_feat = ['CIFAR10', 'SVHN', 'Food101']
        self.dataset_feat = ['CIFAR10', 'MNIST', 'SVHN', 'Textures', 'Places365']
        # self.dataset_ID = 'KMNIST'
        # self.dataset_feat = ['MNIST', 'FMNIST', 'KMNIST', 'Letters']
        self.case = '15d'
        self.methods = ['NCM', 'KNN', 'FKM', 'CKM']
        
        # Model parameters
        self.model_type = 'spike'
        self.resnet_model = 18
        self.num_classes = 10
        self.expansion = 10
        self.num_time_steps_train = 1
        self.num_time_steps_extract = 1
        self.override_feature_extraction = False

        # Training parameters
        self.batch_size = 32
        self.epochs = 200
        self.full_train = True
        self.auto_aug = False
        self.pretrained = False
        self.loss = 'mse_count_loss'
        self.fit = 'spike'

        # Set the "test" mode for the pipeline: train, test, feature extraction, statistics
        # set the "test_population" mode for the pipeline: train, test, feature extraction for different extraction time steps, statistics
        self.mode = 'test_population'

        # Parser defaults (only used if explicitly provided)
        self.seed = 42
        self.device = 'cuda'
        
        self.num_workers = 4
        self.learning_rate = 2e-4
        self.optimizer = 'adam'
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.scheduler = 'cosine'
        self.gradient_clipping = 0.1
    
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
        if 'time_steps_train' in provided_args:
            self.num_time_steps_train = args.time_steps_train
        if  'time_steps_extract' in provided_args:
            self.num_time_steps_extract = args.time_steps_extract
        if 'expansion' in provided_args:
            self.expansion = args.expansion
        if 'auto_aug' in provided_args:
            self.auto_aug = args.auto_aug
        if 'population_coding' in provided_args:
            self.population_coding = args.population_coding

