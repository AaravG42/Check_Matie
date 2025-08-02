import torch
import numpy as np
import struct
import argparse
from pathlib import Path

def convert_pytorch_to_binary(pytorch_file, output_file):
    print(f"Loading PyTorch model from {pytorch_file}")
    
    checkpoint = torch.load(pytorch_file, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    hidden_size = state_dict['input_weights.weight'].shape[0]
    input_size = state_dict['input_weights.weight'].shape[1]
    
    if input_size != 768:
        raise ValueError(f"Expected input size 768, got {input_size}")
    
    print(f"Model configuration: hidden_size={hidden_size}, input_size={input_size}")
    
    QA = 255
    QB = 64
    
    input_weights = state_dict['input_weights.weight'].transpose(0, 1)
    input_weights_q = torch.round(input_weights * QA).clamp(-32768, 32767).to(torch.int16)
    
    input_bias = state_dict['input_weights.bias']
    input_bias_q = torch.round(input_bias * QA).clamp(-32768, 32767).to(torch.int16)
    
    output_weights = state_dict['output_layer.weight'].squeeze(0)
    output_weights_q = torch.round(output_weights * QB).clamp(-32768, 32767).to(torch.int16)
    
    output_bias = state_dict['output_layer.bias'].item()
    output_bias_q = int(round(output_bias * QA * QB))
    output_bias_q = max(-2147483648, min(2147483647, output_bias_q))
    
    with open(output_file, 'wb') as f:
        f.write(struct.pack('<i', hidden_size))
        f.write(input_weights_q.numpy().tobytes())
        f.write(input_bias_q.numpy().tobytes())
        f.write(output_weights_q.numpy().tobytes())
        f.write(struct.pack('<i', output_bias_q))
    
    print(f"Binary weights saved to {output_file}")

def verify_binary_file(binary_file):
    print(f"Verifying binary file: {binary_file}")
    
    with open(binary_file, 'rb') as f:
        hidden_size = struct.unpack('<i', f.read(4))[0]
        print(f"Hidden size: {hidden_size}")
        
        input_weights_size = 768 * hidden_size
        input_weights_bytes = f.read(input_weights_size * 2)
        input_weights = np.frombuffer(input_weights_bytes, dtype=np.int16)
        input_weights = input_weights.reshape(768, hidden_size)
        print(f"Input weights shape: {input_weights.shape}")
        
        input_bias_bytes = f.read(hidden_size * 2)
        input_bias = np.frombuffer(input_bias_bytes, dtype=np.int16)
        print(f"Input bias shape: {input_bias.shape}")
        
        output_weights_bytes = f.read(2 * hidden_size * 2)
        output_weights = np.frombuffer(output_weights_bytes, dtype=np.int16)
        print(f"Output weights shape: {output_weights.shape}")
        
        output_bias = struct.unpack('<i', f.read(4))[0]
        print(f"Output bias: {output_bias}")
        
        remaining = f.read()
        if remaining:
            print(f"Warning: {len(remaining)} bytes remaining in file")
        else:
            print("File verification successful!")

def main():
    parser = argparse.ArgumentParser(description='Convert NNUE weights to binary format')
    parser.add_argument('input', help='Input PyTorch model file (.pth)')
    parser.add_argument('output', help='Output binary file (.bin)')
    parser.add_argument('--verify', action='store_true', help='Verify output file')
    
    args = parser.parse_args()
    
    try:
        convert_pytorch_to_binary(args.input, args.output)
        
        if args.verify:
            verify_binary_file(args.output)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 