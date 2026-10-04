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
    # Case 17e trains on mse and E = 25, that is population coding

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

    # Case 30:
    # InDistribution: CIFAR10
    # OutOfDistribution: 'MNIST', 'SVHN', 'Textures', 'Places
    # Auto augmentation: A = False
    # mse_loss
    # Epochs = 200
    # Test_1


from email import parser
import yaml
from pathlib import Path
from utils.snn_loss import resolve_ce_options


def load_config_from_yaml(yaml_path):
    """
    Load configuration from a YAML file and return an ExperimentConfig object.
    
    Args:
        yaml_path (str): Path to the YAML configuration file
        
    Returns:
        ExperimentConfig: Configured experiment object
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        raise ValueError(f"Configuration file is empty: {yaml_path}")
    
    config = ExperimentConfig()
    
    # Load dataset parameters
    if 'dataset' in config_dict:
        dataset_cfg = config_dict['dataset']
        if 'id' in dataset_cfg:
            config.dataset_ID = dataset_cfg['id']
        if 'features' in dataset_cfg:
            config.dataset_feat = dataset_cfg['features']
    
    # Load model parameters
    if 'model' in config_dict:
        model_cfg = config_dict['model']
        if 'type' in model_cfg:
            config.model_type = model_cfg['type']
        if 'resnet_model' in model_cfg:
            config.resnet_model = model_cfg['resnet_model']
        if 'num_classes' in model_cfg:
            config.num_classes = model_cfg['num_classes']
        if 'expansion' in model_cfg:
            config.expansion = model_cfg['expansion']
        if 'pooling' in model_cfg:
            config.pooling = model_cfg['pooling']
        if 'readout' in model_cfg:
            config.readout = model_cfg['readout']
        if 'num_time_steps_train' in model_cfg:
            config.num_time_steps_train = model_cfg['num_time_steps_train']
        if 'num_time_steps_extract' in model_cfg:
            config.num_time_steps_extract = model_cfg['num_time_steps_extract']
        if 'seed' in model_cfg:
            config.seed = model_cfg['seed']
    
    # Load training parameters
    if 'training' in config_dict:
        training_cfg = config_dict['training']
        if 'batch_size' in training_cfg:
            config.batch_size = training_cfg['batch_size']
        if 'full_train' in training_cfg:
            config.full_train = training_cfg['full_train']
        if 'auto_aug' in training_cfg:
            config.auto_aug = training_cfg['auto_aug']
        if 'pretrained' in training_cfg:
            config.pretrained = training_cfg['pretrained']
        if 'epochs' in training_cfg:
            config.epochs = training_cfg['epochs']
        elif config.auto_aug:
            # If epochs not specified, default based on auto_aug
            config.epochs = 400 if config.auto_aug else 200
        if 'loss' in training_cfg:
            config.loss = training_cfg['loss']
        if 'fit' in training_cfg:
            config.fit = training_cfg['fit']
        if 'ce_source' in training_cfg:
            config.ce_source = training_cfg['ce_source']
        if 'ce_mode' in training_cfg:
            config.ce_mode = training_cfg['ce_mode']
        if 'population_reduction' in training_cfg:
            config.population_reduction = training_cfg['population_reduction']

        config.ce_source, config.ce_mode, config.population_reduction = resolve_ce_options(
            config.ce_source,
            config.ce_mode,
            config.population_reduction,
            config.fit,
        )
    
    # Load mode
    if 'mode' in config_dict:
        config.mode = config_dict['mode']
    if 'test_type' in config_dict:
        config.test_type = config_dict['test_type']
    
    # Load device and processing
    if 'device' in config_dict:
        config.device = config_dict['device']
    if 'num_workers' in config_dict:
        config.num_workers = config_dict['num_workers']
    
    # Load optimizer parameters
    if 'optimizer' in config_dict:
        optimizer_cfg = config_dict['optimizer']
        if 'name' in optimizer_cfg:
            config.optimizer = optimizer_cfg['name']
        if 'learning_rate' in optimizer_cfg:
            config.learning_rate = optimizer_cfg['learning_rate']
        if 'momentum' in optimizer_cfg:
            config.momentum = optimizer_cfg['momentum']
        if 'weight_decay' in optimizer_cfg:
            config.weight_decay = optimizer_cfg['weight_decay']
    
    # Load scheduler parameters
    if 'scheduler' in config_dict:
        scheduler_cfg = config_dict['scheduler']
        if 'name' in scheduler_cfg:
            config.scheduler = scheduler_cfg['name']
        if 'step_size' in scheduler_cfg:
            config.step_size = scheduler_cfg['step_size']
    
    # Load gradient clipping
    if 'gradient_clipping' in config_dict:
        config.gradient_clipping = config_dict['gradient_clipping']
    
    # Load experiment-specific parameters
    if 'experiment' in config_dict:
        exp_cfg = config_dict['experiment']
        if 'case' in exp_cfg:
            config.case = exp_cfg['case']
        if 'methods' in exp_cfg:
            config.methods = exp_cfg['methods']
        if 'methods_1' in exp_cfg:
            config.methods_1 = exp_cfg['methods_1']
        if 'methods_2' in exp_cfg:
            config.methods_2 = exp_cfg['methods_2']
        if 'seeds' in exp_cfg:
            config.seeds = exp_cfg['seeds']
        if 'expansions' in exp_cfg:
            config.expansions = exp_cfg['expansions']
        if 'resnet_models' in exp_cfg:
            config.resnet_models = exp_cfg['resnet_models']
        if 'override_feature_extraction' in exp_cfg:
            config.override_feature_extraction = exp_cfg['override_feature_extraction']
        if 'near_ood' in exp_cfg:
            config.near_ood = exp_cfg['near_ood']
        if 'far_ood' in exp_cfg:
            config.far_ood = exp_cfg['far_ood']
    
    return config


class ExperimentConfig:
    def __init__(self):
        # Dataset parameters
        self.dataset_ID = 'CIFAR10'
        self.dataset_feat = ['CIFAR10', 'CIFAR100', 'tImage200']
        
        # Model parameters
        self.model_type = 'spike'
        self.resnet_model = 18
        self.num_classes = 10
        self.expansion = 1
        self.pooling = 'max'
        self.readout = 'lif'
        self.num_time_steps_train = 1
        self.num_time_steps_extract = 1
        self.seed = 42

        # Training parameters
        self.batch_size = 64
        self.full_train = True
        self.auto_aug = True
        self.pretrained = False
        self.epochs = 400  # Default for auto_aug=True
        self.loss = 'count_loss'
        self.fit = 'spike'
        self.ce_source = None
        self.ce_mode = None
        self.population_reduction = None
        self.checkpointPeriod = 10

        # Pipeline mode
        self.mode = 'test'
        self.test_type = 'experiment_1'  # Options: 'standard', 'population', 'single_step', 'accuracy', 'experiment_1', 'experiment_2', 'experiment_3'

        # Device and processing
        self.device = 'cuda'
        self.num_workers = 4
        
        # Optimizer parameters
        self.learning_rate = 2e-4
        self.optimizer = 'adam'
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        # Scheduler parameters
        self.scheduler = 'cosine'
        self.step_size = None
        self.gradient_clipping = 0.1
        
        # Experiment sweep parameters (for multi_train, multi_test, ex_1, ex_2)
        # These are None by default and will use hardcoded defaults in main.py if not specified
        self.case = '42'
        self.methods = None
        self.methods_1 = None
        self.methods_2 = None
        self.seeds = None
        self.expansions = None
        self.resnet_models = None
        self.override_feature_extraction = False
        self.near_ood = None
        self.far_ood = None

