import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NNUE(nn.Module):
    def __init__(self, hidden_size=256):
        super(NNUE, self).__init__()
        
        self.INPUT_SIZE = 768  # 6 pieces * 2 colors * 64 squares
        self.HIDDEN_SIZE = hidden_size
        self.SCALE = 400
        self.QA = 255  # Quantization factor for accumulator
        self.QB = 64   # Quantization factor for output weights
        
        # Input to accumulator weights (shared between perspectives)
        self.input_weights = nn.Linear(self.INPUT_SIZE, self.HIDDEN_SIZE, bias=True)
        
        # Output layer weights (takes concatenated accumulators)
        self.output_layer = nn.Linear(2 * self.HIDDEN_SIZE, 1, bias=True)
        
        # Initialize weights
        nn.init.uniform_(self.input_weights.weight, -0.1, 0.1)
        nn.init.uniform_(self.input_weights.bias, -1.0, 1.0)
        nn.init.uniform_(self.output_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.output_layer.bias, -1.0, 1.0)

    def encode_position(self, board_state):
        white_features = torch.zeros(self.INPUT_SIZE, dtype=torch.float32)
        black_features = torch.zeros(self.INPUT_SIZE, dtype=torch.float32)
        
        for square, (piece_type, color) in board_state.items():
            # White perspective
            white_index = color * 64 * 6 + piece_type * 64 + square
            white_features[white_index] = 1.0
            
            # Black perspective (flip square and color)
            black_square = square ^ 56  # Vertical flip
            black_color = 1 - color     # Flip color
            black_index = black_color * 64 * 6 + piece_type * 64 + black_square
            black_features[black_index] = 1.0
        
        return white_features, black_features

    def forward(self, white_features, black_features, side_to_move):
        batch_size = white_features.shape[0]
        
        # Compute accumulators for both perspectives
        white_accumulator = self.input_weights(white_features)
        black_accumulator = self.input_weights(black_features)
        
        # Apply SCReLU activation (Squared Clipped ReLU)
        white_activated = self._screlu(white_accumulator)
        black_activated = self._screlu(black_accumulator)
        
        # Concatenate accumulators based on side to move
        stm_accumulator = torch.where(
            side_to_move.unsqueeze(1) == 0,
            white_activated,
            black_activated
        )
        nstm_accumulator = torch.where(
            side_to_move.unsqueeze(1) == 0,
            black_activated,
            white_activated
        )
        
        # Concatenate accumulators
        combined_features = torch.cat([stm_accumulator, nstm_accumulator], dim=1)
        
        # Output layer
        output = self.output_layer(combined_features)
        
        # Scale output to centipawn range
        output = output * self.SCALE
        
        return output
    
    def _screlu(self, x):
        clipped = torch.clamp(x, 0, 1)
        return clipped * clipped
    
    def get_quantized_weights(self):
        # Quantize input weights and biases
        input_weights_q = torch.round(self.input_weights.weight.data * self.QA).clamp(-32768, 32767).to(torch.int16)
        input_bias_q = torch.round(self.input_weights.bias.data * self.QA).clamp(-32768, 32767).to(torch.int16)
        
        # Quantize output weights and bias
        output_weights_q = torch.round(self.output_layer.weight.data * self.QB).clamp(-32768, 32767).to(torch.int16)
        output_bias_q = torch.round(self.output_layer.bias.data * (self.QA * self.QB)).clamp(-2147483648, 2147483647).to(torch.int32)
        
        return {
            'input_weights': input_weights_q,
            'input_bias': input_bias_q,
            'output_weights': output_weights_q,
            'output_bias': output_bias_q.item(),
            'hidden_size': self.HIDDEN_SIZE
        }


class NNUELoss(nn.Module):
    def __init__(self, lambda_eval=1.0, lambda_result=0.0):
        super(NNUELoss, self).__init__()
        self.lambda_eval = lambda_eval
        self.lambda_result = lambda_result
    
    def forward(self, predictions, eval_targets, result_targets=None):
        eval_loss = F.mse_loss(predictions, eval_targets)
        total_loss = self.lambda_eval * eval_loss
        
        if result_targets is not None and self.lambda_result > 0:
            pred_probs = torch.sigmoid(predictions / 400.0)
            result_loss = F.binary_cross_entropy(pred_probs, result_targets)
            total_loss += self.lambda_result * result_loss
        
        return total_loss 