from utils.Utils import Utils
from snntorch import spikegen
from torch.utils.data import DataLoader

# Test train posibilities
dataSet = 'MNIST'

dataset_train, dataset_test = Utils.load_data(dataSet)

# Get the image size
print("Get image size ...")
train_tensor, train_label = dataset_train[0]
imageSize = train_tensor.size()
print(f'Image size: {imageSize[0]}, {imageSize[1]}, {imageSize[2]}')
inputSize = imageSize[0] * imageSize[1] * imageSize[2]

# Batch size
batchSize = 16

# Number of steps
num_steps = 50

train_data = dataset_train
test_data = dataset_test
print("Train data length ", len(train_data))
print("Test data length ", len(test_data))

train_loader = DataLoader(train_data, batchSize, shuffle=True)
test_loader = DataLoader(test_data, batchSize)

# make minibatches
# iterate trought minibatches 


# Iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)

print(spike_data.size())