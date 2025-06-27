#!/bin/bash

# Systematic test script for MyEngine vs Stockfish using cutechess-cli
# Usage: ./test_vs_stockfish.sh [games] [time_control] [stockfish_skill]
# Example: ./test_vs_stockfish.sh 20 5+0 8

ENGINE_PATH="./chess_engine"
STOCKFISH_PATH="stockfish"  # Change if not in PATH
CUTECHESS="cutechess-cli"    # Change if not in PATH

GAMES="${1:-20}"
TC="${2:-5+0}"
SKILL="${3:-8}"
PGN_FILE="test_vs_stockfish_$(date +%Y%m%d_%H%M%S).pgn"
LOG_FILE="test_vs_stockfish_$(date +%Y%m%d_%H%M%S).log"

# Print configuration
echo "========================================="
echo "Testing MyEngine vs Stockfish"
echo "Games:         $GAMES"
echo "Time control:  $TC"
echo "Stockfish skill: $ELO"
echo "PGN output:    $PGN_FILE"
echo "========================================="

# Run the match
$CUTECHESS \
  -engine cmd="$ENGINE_PATH" name="MyEngine" \
  -engine cmd="$STOCKFISH_PATH" name="Stockfish" option.UCI_LimitStrength=true option.UCI_Elo=$SKILL \
  -each proto=uci tc=$TC \
  -games $GAMES \
  -concurrency 2 \
  -draw movenumber=50 movecount=5 score=5 \
  -resign movecount=3 score=800 \
  -pgnout "$PGN_FILE" | tee "$LOG_FILE"

echo "========================================="
echo "Match complete!"
echo "Results summary:"
grep -E 'Score of|1-0|0-1|1/2-1/2' "$LOG_FILE"
echo "PGN saved to $PGN_FILE"
echo "=========================================" 