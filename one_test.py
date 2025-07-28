import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

def test_sequential_behavior():
    print("Testing nn.Sequential behavior with LIF layers...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    input_size = 784
    hidden_size = 128
    output_size = 500
    
    beta = 0.9
    grad = surrogate.fast_sigmoid()
    
    # Test input
    test_input = torch.randn(batch_size, input_size).to(device)
    
    # Create individual layers to test step by step
    flatten = nn.Flatten()
    linear1 = nn.Linear(input_size, hidden_size)
    lif1 = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)
    linear2 = nn.Linear(hidden_size, output_size)
    lif2 = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
    
    # Move to device
    linear1 = linear1.to(device)
    lif1 = lif1.to(device)
    linear2 = linear2.to(device)
    lif2 = lif2.to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    # Step 1: Flatten
    out1 = flatten(test_input)
    print(f"After flatten: {out1.shape}, type: {type(out1)}")
    
    # Step 2: First linear
    out2 = linear1(out1)
    print(f"After linear1: {out2.shape}, type: {type(out2)}")
    
    # Step 3: First LIF (this is where it gets interesting!)
    out3 = lif1(out2)
    print(f"After lif1: {type(out3)}")
    if isinstance(out3, tuple):
        print(f"  Tuple length: {len(out3)}")
        for i, elem in enumerate(out3):
            print(f"  Element {i}: shape={elem.shape}")
    else:
        print(f"  Tensor shape: {out3.shape}")
    
    # Step 4: Second linear (THE CRITICAL STEP!)
    try:
        out4 = linear2(out3)
        print(f"After linear2: {type(out4)}")
        if isinstance(out4, tuple):
            print(f"  Tuple length: {len(out4)}")
            for i, elem in enumerate(out4):
                print(f"  Element {i}: shape={elem.shape}")
        else:
            print(f"  Tensor shape: {out4.shape}")
    except Exception as e:
        print(f"Error at linear2: {e}")
        print(f"Trying to pass only first element of tuple...")
        if isinstance(out3, tuple):
            out4 = linear2(out3[0])  # Pass only spikes
            print(f"After linear2 (spikes only): {out4.shape}")
    
    # Step 5: Final LIF
    try:
        out5 = lif2(out4)
        print(f"After lif2: {type(out5)}")
        if isinstance(out5, tuple):
            print(f"  Final tuple length: {len(out5)}")
            for i, elem in enumerate(out5):
                print(f"  Element {i}: shape={elem.shape}")
        else:
            print(f"  Final tensor shape: {out5.shape}")
    except Exception as e:
        print(f"Error at lif2: {e}")
    
    print("\n" + "="*50)
    
    # Now test the full Sequential
    print("Testing full nn.Sequential...")
    net_pop = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_size),
        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
        nn.Linear(hidden_size, output_size),
        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
    ).to(device)
    
    try:
        final_output = net_pop(test_input)
        print(f"Sequential output type: {type(final_output)}")
        if isinstance(final_output, tuple):
            print(f"Sequential tuple length: {len(final_output)}")
            for i, elem in enumerate(final_output):
                print(f"  Element {i}: shape={elem.shape}")
        else:
            print(f"Sequential tensor shape: {final_output.shape}")
    except Exception as e:
        print(f"Sequential error: {e}")

if __name__ == "__main__":
    test_sequential_behavior()