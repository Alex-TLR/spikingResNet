import torch, torch.nn as nn
import snntorch as snn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import surrogate
from snntorch import utils
import snntorch.functional as SF
from snntorch import backprop

def setup_data(batch_size=128, data_path='/tmp/data/fmnist'):
    """Setup data loaders for Fashion-MNIST"""
    
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    fmnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

def create_networks(device, num_inputs=28*28, num_hidden=128, num_outputs=10, pop_outputs=500):
    """Create both standard and population coding networks"""
    
    # spiking neuron parameters
    beta = 0.9  # neuron decay rate 
    grad = surrogate.fast_sigmoid()

    # Standard network
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(num_inputs, num_hidden),
                        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                        nn.Linear(num_hidden, num_outputs),
                        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
                        ).to(device)

    # Population coding network
    net_pop = nn.Sequential(nn.Flatten(),
                            nn.Linear(num_inputs, num_hidden),
                            snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                            nn.Linear(num_hidden, pop_outputs),
                            snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
                            ).to(device)
    
    return net, net_pop

def check_model_outputs(device):
    """Check what outputs the models return"""
    
    print("="*60)
    print("CHECKING MODEL OUTPUT SHAPES")
    print("="*60)
    
    # Create test data
    batch_size = 4
    test_input = torch.randn(batch_size, 28*28).to(device)
    print(f"Test input shape: {test_input.shape}")
    
    # Create networks
    net, net_pop = create_networks(device)
    
    print("\n--- Standard Network (net) ---")
    try:
        with torch.no_grad():
            utils.reset(net)
            output_net = net(test_input)
            
            print(f"Output type: {type(output_net)}")
            if isinstance(output_net, tuple):
                print(f"Tuple length: {len(output_net)}")
                for i, elem in enumerate(output_net):
                    print(f"  Element {i}: type={type(elem)}, shape={elem.shape if hasattr(elem, 'shape') else 'No shape'}")
            else:
                print(f"Output shape: {output_net.shape}")
                
    except Exception as e:
        print(f"Standard network error: {e}")
    
    print("\n--- Population Network (net_pop) ---")
    try:
        with torch.no_grad():
            utils.reset(net_pop)
            output_pop = net_pop(test_input)
            
            print(f"Output type: {type(output_pop)}")
            if isinstance(output_pop, tuple):
                print(f"Tuple length: {len(output_pop)}")
                for i, elem in enumerate(output_pop):
                    print(f"  Element {i}: type={type(elem)}, shape={elem.shape if hasattr(elem, 'shape') else 'No shape'}")
            else:
                print(f"Output shape: {output_pop.shape}")
                
    except Exception as e:
        print(f"Population network error: {e}")
    
    print("\n--- Testing with BPTT Interface ---")
    try:
        with torch.no_grad():
            utils.reset(net_pop)
            # This is what BPTT expects internally
            spk, mem = net_pop(test_input)
            print(f"BPTT spk shape: {spk.shape}")
            print(f"BPTT mem shape: {mem.shape}")
            print("✅ BPTT interface works correctly")
            
    except Exception as e:
        print(f"❌ BPTT interface error: {e}")
        
        # Try alternative unpacking
        try:
            with torch.no_grad():
                utils.reset(net_pop)
                output = net_pop(test_input)
                if isinstance(output, tuple) and len(output) == 2:
                    spk, mem = output
                    print(f"Alternative spk shape: {spk.shape}")
                    print(f"Alternative mem shape: {mem.shape}")
                    print("✅ Alternative unpacking works")
                else:
                    print(f"❌ Cannot unpack output: {type(output)}")
        except Exception as e2:
            print(f"❌ Alternative unpacking error: {e2}")
    
    print("="*60)

def test_accuracy(data_loader, net, num_steps, device, population_code=False, num_classes=10):
    """Test accuracy function for both standard and population coding"""
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        data_loader = iter(data_loader)
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            utils.reset(net)
            spk_rec, _ = net(data)

            if population_code:
                acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets, population_code=True, num_classes=num_classes) * spk_rec.size(1)
            else:
                acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets) * spk_rec.size(1)
                
            total += spk_rec.size(1)

    return acc/total

def train_standard_network(net, train_loader, test_loader, device, num_epochs=5, num_steps=1):
    """Train standard SNN network"""
    print("Training Standard SNN Network...")
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0)
    
    # Training loop
    for epoch in range(num_epochs):
        avg_loss = backprop.BPTT(net, train_loader, num_steps=num_steps,
                              optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)
        
        test_acc = test_accuracy(test_loader, net, num_steps, device, population_code=False)
        
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Test set accuracy: {test_acc*100:.3f}%\n")

def train_population_network(net_pop, train_loader, test_loader, device, num_epochs=5, num_steps=1, num_classes=10):
    """Train population coding SNN network"""
    print("Training Population Coding SNN Network...")
    
    # Setup optimizer and loss for population coding
    optimizer = torch.optim.Adam(net_pop.parameters(), lr=2e-3, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=num_classes)
    
    # Training loop
    for epoch in range(num_epochs):
        avg_loss = backprop.BPTT(net_pop, train_loader, num_steps=num_steps,
                              optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)
        
        test_acc = test_accuracy(test_loader, net_pop, num_steps, device, population_code=True, num_classes=num_classes)
        
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Test set accuracy: {test_acc*100:.3f}%\n")

def analyze_bptt_behavior(device, num_steps=4):
    """Analyze what BPTT does with multi-step outputs"""
    
    print("="*60)
    print(f"ANALYZING BPTT BEHAVIOR WITH num_steps={num_steps}")
    print("="*60)
    
    # Create test data
    batch_size = 4
    test_input = torch.randn(batch_size, 28*28).to(device)
    test_targets = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Create networks
    net, net_pop = create_networks(device)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test targets shape: {test_targets.shape}")
    
    # Test what net_pop returns in a single forward pass
    print("\n--- Single Forward Pass ---")
    with torch.no_grad():
        utils.reset(net_pop)
        output = net_pop(test_input)
        if isinstance(output, tuple):
            spk, mem = output
            print(f"Single pass spk shape: {spk.shape}")
            print(f"Single pass mem shape: {mem.shape}")
    
    # Simulate what BPTT does internally with multiple time steps
    print(f"\n--- Simulating BPTT with {num_steps} steps ---")
    spk_rec = []
    mem_rec = []
    
    with torch.no_grad():
        utils.reset(net_pop)
        
        for step in range(num_steps):
            # This is what BPTT does: calls the network multiple times
            spk, mem = net_pop(test_input)  # Same input each time step
            spk_rec.append(spk)
            mem_rec.append(mem)
            print(f"Step {step}: spk shape {spk.shape}, mem shape {mem.shape}")
    
    # Stack like BPTT does
    spk_stacked = torch.stack(spk_rec, dim=0)  # [T, B, features]
    mem_stacked = torch.stack(mem_rec, dim=0)  # [T, B, features]
    
    print(f"\nAfter stacking:")
    print(f"spk_stacked shape: {spk_stacked.shape}")
    print(f"mem_stacked shape: {mem_stacked.shape}")
    
    # Test the loss function with multi-step data
    print(f"\n--- Testing Loss Function ---")
    
    # Setup loss function
    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, 
                               population_code=True, num_classes=10)
    
    try:
        # Test with single step output [B, features]
        loss_single = loss_fn(spk, test_targets)
        print(f"✅ Single step loss: {loss_single.item():.4f}")
        print(f"   Input shape: {spk.shape}, Target shape: {test_targets.shape}")
    except Exception as e:
        print(f"❌ Single step loss error: {e}")
    
    try:
        # Test with multi-step output [T, B, features]
        loss_multi = loss_fn(spk_stacked, test_targets)
        print(f"✅ Multi-step loss: {loss_multi.item():.4f}")
        print(f"   Input shape: {spk_stacked.shape}, Target shape: {test_targets.shape}")
    except Exception as e:
        print(f"❌ Multi-step loss error: {e}")
    
    # Test what happens when we average over time
    try:
        spk_avg = spk_stacked.mean(dim=0)  # [B, features]
        loss_avg = loss_fn(spk_avg, test_targets)
        print(f"✅ Time-averaged loss: {loss_avg.item():.4f}")
        print(f"   Input shape: {spk_avg.shape}, Target shape: {test_targets.shape}")
    except Exception as e:
        print(f"❌ Time-averaged loss error: {e}")
    
    print("="*60)

def main():
    """Main function to run the training experiments"""
    
    # Setup parameters
    batch_size = 128
    data_path = '/tmp/data/fmnist'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device("cpu")
    
    # Network parameters
    num_inputs = 28*28
    num_hidden = 128
    num_outputs = 10
    pop_outputs = 500
    num_steps = 4
    num_epochs = 5
    num_classes = 10
    
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Time steps: {num_steps}")
    
    # ✅ NEW: Check model outputs first
    check_model_outputs(device)
    
    print("="*50)

    # Add this analysis
    analyze_bptt_behavior(device, num_steps=num_steps)
    
    # Setup data
    train_loader, test_loader = setup_data(batch_size, data_path)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("="*50)
    
    # Create networks
    net, net_pop = create_networks(device, num_inputs, num_hidden, num_outputs, pop_outputs)
    
    # Train standard network
    train_standard_network(net, train_loader, test_loader, device, num_epochs, num_steps)
    
    print("="*50)
    
    # Train population coding network
    train_population_network(net_pop, train_loader, test_loader, device, num_epochs, num_steps, num_classes)
    
    print("Training completed!")

if __name__ == "__main__":
    main()