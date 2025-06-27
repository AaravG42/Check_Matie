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

## Things I Implemented

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
    - https://www.youtube.com/watch?v=U4ogK0MIzqk
    - https://www.youtube.com/watch?v=_vqlIPDR2TU 
