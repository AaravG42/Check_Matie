import chess
import chess.engine
import numpy as np
import h5py
import random
import argparse
import os
from tqdm import tqdm
import concurrent.futures

def encode_position(board):
    white_features = np.zeros(768, dtype=np.float32)
    black_features = np.zeros(768, dtype=np.float32)
    
    piece_map = board.piece_map()
    
    for square, piece in piece_map.items():
        piece_type = piece.piece_type - 1  # 0-5 for pawn to king
        color = 0 if piece.color else 1    # 0 for white, 1 for black
        
        # White perspective
        white_index = color * 384 + piece_type * 64 + square
        white_features[white_index] = 1.0
        
        # Black perspective (flip square and color)
        black_square = square ^ 56  # Vertical flip
        black_color = 1 - color     # Flip color
        black_index = black_color * 384 + piece_type * 64 + black_square
        black_features[black_index] = 1.0
    
    return white_features, black_features

def generate_random_position(move_count=30):
    board = chess.Board()
    moves = []
    
    for _ in range(move_count):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
            
        move = random.choice(legal_moves)
        board.push(move)
        moves.append(move)
        
        # Stop if game is over
        if board.is_game_over():
            break
    
    return board, moves

def evaluate_position(board, engine, depth=12):
    result = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = result["score"].white().score(mate_score=30000)
    return score

def generate_training_data(engine_path, num_positions, output_file, num_workers=4):
    print(f"Generating {num_positions} training positions using {num_workers} workers")
    
    # Initialize arrays to store data
    white_features_list = []
    black_features_list = []
    side_to_move_list = []
    evaluations_list = []
    
    # Function to generate a single position
    def generate_position():
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            board, _ = generate_random_position(random.randint(5, 60))
            
            # Evaluate position
            try:
                evaluation = evaluate_position(board, engine)
                
                # Encode position
                white_features, black_features = encode_position(board)
                side_to_move = 0 if board.turn else 1
                
                return white_features, black_features, side_to_move, evaluation
            except Exception as e:
                print(f"Error evaluating position: {e}")
                return None
    
    # Generate positions using multiple workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_position) for _ in range(num_positions)]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_positions):
            result = future.result()
            if result:
                white_features, black_features, side_to_move, evaluation = result
                white_features_list.append(white_features)
                black_features_list.append(black_features)
                side_to_move_list.append(side_to_move)
                evaluations_list.append(evaluation)
    
    # Convert lists to arrays
    white_features_array = np.array(white_features_list)
    black_features_array = np.array(black_features_list)
    side_to_move_array = np.array(side_to_move_list)
    evaluations_array = np.array(evaluations_list)
    
    # Save to HDF5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('white_features', data=white_features_array)
        f.create_dataset('black_features', data=black_features_array)
        f.create_dataset('side_to_move', data=side_to_move_array)
        f.create_dataset('evaluations', data=evaluations_array)
    
    print(f"Generated {len(white_features_list)} positions")
    print(f"Data saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate NNUE training data')
    parser.add_argument('--engine', type=str, required=True, help='Path to UCI chess engine')
    parser.add_argument('--positions', type=int, default=10000, help='Number of positions to generate')
    parser.add_argument('--output', type=str, default='data/training_data.h5', help='Output file path')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    generate_training_data(args.engine, args.positions, args.output, args.workers)

if __name__ == '__main__':
    main() 