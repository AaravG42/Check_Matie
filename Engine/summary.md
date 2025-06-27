Optimisations over alpha-beta pruning:
(not really an optimisation) changed from minmax to negamax
transposition tables
    zobrist hashing
    each entry stores - key (zobrist hash), best move, depth, score, flag (exact, alpha, beta - can be used to prune branches even if exact value isn't known)
    16mb
    depth-preferred replacement
iterative deepening
    keeps doing iterative deepening and checks if time limit is reached every 1024 nodes
quiescence search
    for leaf nodes (that do not cause cutoff), it is like the negamax search but only considers capture moves (max depth limited to 10)
    avoids the horizon effect

Evaluation function:
material: pawn 100, knight 320, bishop 330, rook 500, queen 900
piece-square tables: middlegame and endgame for pawns/kings
pawns: +10 each
mobility: +5 per legal move
in check: -20
endgame (<=6 pieces left):
    +10*(distance of opp king from center)
    +10*(14-distance bt kings)

Move Ordering:
move given max pref. if it is already best move decided by a previous lower depth search and in the transposition table (due to iterative deepening)
captures are given next pref. with Most Valuable Victim - Least Valuable Attacker heuristic
next promotions then checks