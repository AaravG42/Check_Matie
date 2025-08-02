import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import os
import argparse
from nnue_model import NNUE, NNUELoss

class NNUEDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.length = f['white_features'].shape[0]
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            white_features = torch.tensor(f['white_features'][idx], dtype=torch.float32)
            black_features = torch.tensor(f['black_features'][idx], dtype=torch.float32)
            side_to_move = torch.tensor(f['side_to_move'][idx], dtype=torch.float32)
            evaluation = torch.tensor(f['evaluations'][idx], dtype=torch.float32)
            
        return white_features, black_features, side_to_move, evaluation

def train_nnue(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_dataset = NNUEDataset(args.data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    print(f"Loaded {len(train_dataset)} training positions")
    
    # Create model
    model = NNUE(hidden_size=args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = NNUELoss(lambda_eval=1.0)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for white_features, black_features, side_to_move, targets in progress_bar:
            white_features = white_features.to(device)
            black_features = black_features.to(device)
            side_to_move = side_to_move.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(white_features, black_features, side_to_move)
            loss = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.3f}'})
        
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.3f}")
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'nnue_best.pth'))
            print(f"Saved best model with loss {best_loss:.3f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'nnue_final.pth'))
    print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train NNUE model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data (h5 file)')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    
    args = parser.parse_args()
    train_nnue(args)

if __name__ == '__main__':
    main() 