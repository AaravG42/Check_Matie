# ♟️ Chess Engine Project: Classic AI + RL

Build your own competitive chess engine in 8 weeks! This project combines **classical AI** (like Minimax and Alpha-Beta Pruning) with **Reinforcement Learning** to create a fully functional engine you can play against.

> 🎯 Final showdown: Engine-vs-Engine tournament with a **prize** for the winner!

## 🚀 What You'll Learn
- Board representation & move generation
- Minimax, Alpha-Beta pruning, heuristics
- Zobrist hashing & transposition tables
- Efficiently Updatable Neural Network (NNUE)
- RL (Q-learning, self-play, policy tuning)

## 📅 Commitment
Expect ~5 hours/week. It's a bit hectic but super rewarding!

## Things I learnt and Implemented - Midterm

### Optimisations over alpha-beta pruning
- Switched from minimax to negamax (simplifies code, same result)
- Transposition tables
    - Zobrist hashing for fast position keys
    - Each entry: key (zobrist hash), best move, depth, score, flag (exact, alpha, beta)
    - 16MB table, depth-preferred replacement
    - Flags allow pruning even if exact value isn't known
- Iterative deepening
    - Repeatedly deepens search, always has best move so far
    - Checks time limit every 1024 nodes
- Quiescence search
    - At leaf nodes, only considers captures (max depth 10)
    - Avoids horizon effect (missing tactics just beyond search depth)
- Null move pruning (skip a move to quickly detect cutoffs in quiet positions)
- Time management: divides remaining time by 20 for each move if not using fixed movetime
- Principal Variation Search (PVS): fast null-window search for all but first move, full search only if needed
- Move ordering
    - TT move first (from previous search)
    - Captures next (Most Valuable Victim - Least Valuable Attacker)
    - Then promotions, then checks
<!-- - UCI protocol support for easy testing against other engines -->

### Evaluation function
- Material: pawn 100, knight 320, bishop 330, rook 500, queen 900
- Piece-square tables: middlegame and endgame for pawns/kings
- Pawns: +10 each
- Mobility: +5 per legal move
- In check: -20
- Endgame (<=6 pieces left):
    - +10 * (distance of opponent king from center)
    - +10 * (14 - distance between kings)
- Always from side to move's perspective

### References and Tools used 
- Chess Programming Wiki: https://www.chessprogramming.org/
- Stockfish: https://github.com/official-stockfish/Stockfish
- UCI protocol: https://www.shredderchess.com/chess-info/features/uci-universal-chess-interface.html
- Cutechess: https://github.com/cutechess/cutechess
- Zobrist hashing: https://en.wikipedia.org/wiki/Zobrist_hashing
- Sebastian League's Videos:
    https://www.youtube.com/watch?v=U4ogK0MIzqk, 
    https://www.youtube.com/watch?v=_vqlIPDR2TU 

## Things I learnt and Implemented - Endterm

### NNUE (Efficiently Updatable Neural Network) Integration
- Basic C++ NNUE evaluator with incremental updates
- 768 input features (6 pieces × 2 colors × 64 squares)
- Simple 2-layer network: 768 → 256 → 1 
- SCReLU activation (just clipped x²)
- 16-bit weight quantization for the C++ engine
- Incremental accumulator updates (add/subtract features on move make/unmake)
- Dual perspective (white/black viewpoints)
- Binary weight file loading

### NNUE Training Setup
- Basic PyTorch training with MSE loss
- Random position generator + Stockfish evaluation for training data
- Standard Adam optimizer with learning rate decay
- Weight converter from PyTorch model to binary format

### Opening Book Integration
- Polyglot book format support (Perfect2023.bin with 700k+ positions)
- Zobrist hashing for fast position lookup in book entries
- Weighted random move selection from book alternatives
- Book move gets priority over search in opening phase
- Binary file parsing with proper endianness handling
- Fallback to search when position not in book

### Engine Configuration
- Config file system for runtime settings
- NNUE enable/disable toggle
- Hash table size configuration  
- Evaluation file path specification
- Easy switching between classical eval and NNUE

### Advanced Evaluation Features
- Lawnmower mate pattern recognition for endgames
- Enhanced king safety evaluation in endings
- Piece coordination bonuses for major pieces
- Checkmate detection with large bonuses
- Smooth transition between opening/middlegame/endgame phases

