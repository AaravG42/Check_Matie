#!/bin/bash

# Systematic test script for MyEngine vs Stockfish using cutechess-cli
# Usage: ./test_vs_stockfish.sh [games] [time_control] [stockfish_elo]
# Example: ./test_vs_stockfish.sh 20 5+0 1500

ENGINE_PATH="./chess_engine"
STOCKFISH_PATH="stockfish"  # Change if not in PATH
CUTECHESS="cutechess-cli"    # Change if not in PATH

GAMES="${1:-20}"
TC="${2:-5+0}"
ELO="${3:-1500}"
PGN_FILE="test_vs_stockfish_$(date +%Y%m%d_%H%M%S).pgn"
LOG_FILE="test_vs_stockfish_$(date +%Y%m%d_%H%M%S).log"

# Check if NNUE weights exist
NNUE_WEIGHTS="nnue_weights.bin"
if [ ! -f "$NNUE_WEIGHTS" ]; then
    echo "Error: NNUE weights file '$NNUE_WEIGHTS' not found!"
    echo "Available NNUE files:"
    ls -la *.bin 2>/dev/null || echo "No .bin files found"
    echo "Cannot continue without NNUE weights. Exiting."
    exit 1
fi

# Print configuration
echo "========================================="
echo "Testing MyEngine vs Stockfish"
echo "Games:         $GAMES"
echo "Time control:  $TC"
echo "Stockfish ELO: $ELO"
echo "Use NNUE:      true"
echo "PGN output:    $PGN_FILE"
echo "========================================="


# Run the match
$CUTECHESS \
  -engine cmd="$ENGINE_PATH" name="MyEngine" option.Use\ NNUE=true option.EvalFile=$NNUE_WEIGHTS \
  -engine cmd="$STOCKFISH_PATH" name="Stockfish" option.UCI_LimitStrength=true option.UCI_Elo=$ELO \
  -each proto=uci tc=$TC \
  -games $GAMES \
  -concurrency 5 \
  -draw movenumber=50 movecount=5 score=5 \
  -resign movecount=3 score=800 \
  -pgnout "$PGN_FILE" | tee "$LOG_FILE"

echo "========================================="
echo "Match complete!"
echo "Results summary:"
grep -E 'Score of|1-0|0-1|1/2-1/2' "$LOG_FILE"
echo "PGN saved to $PGN_FILE"
echo "=========================================" 