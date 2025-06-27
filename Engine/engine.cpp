#include "chess.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <climits>

using namespace chess;

const int MATE_VALUE = 30000;
const int DRAW_VALUE = 0;
const int INF = 32000;

enum TTFlag {
    TT_EXACT = 0,
    TT_ALPHA = 1,
    TT_BETA = 2
};

struct TTEntry {
    std::uint64_t key;
    Move best_move;
    int depth;
    int score;
    TTFlag flag;
    
    TTEntry() : key(0), best_move(Move::NO_MOVE), depth(0), score(0), flag(TT_EXACT) {}
};

class TranspositionTable {
private:
    std::vector<TTEntry> table;
    size_t size_mask;
    
public:
    TranspositionTable(size_t size_mb = 16) {
        size_t size = (size_mb * 1024 * 1024) / sizeof(TTEntry);
        size_t actual_size = 1;
        while (actual_size < size) actual_size <<= 1;
        
        table.resize(actual_size);
        size_mask = actual_size - 1;
    }
    
    void store(std::uint64_t key, Move move, int depth, int score, TTFlag flag) {
        size_t index = key & size_mask;
        TTEntry& entry = table[index];
        
        if (entry.key == 0 || entry.depth <= depth || entry.key == key) {
            entry.key = key;
            entry.best_move = move;
            entry.depth = depth;
            entry.score = score;
            entry.flag = flag;
        }
    }
    
    TTEntry* probe(std::uint64_t key) {
        size_t index = key & size_mask;
        TTEntry& entry = table[index];
        
        if (entry.key == key) {
            return &entry;
        }
        return nullptr;
    }
    
    void clear() {
        std::fill(table.begin(), table.end(), TTEntry());
    }
};

class ChessEngine {
private:
    Board board;
    TranspositionTable tt;
    bool stop_search;
    int nodes_searched;
    std::chrono::steady_clock::time_point search_start;
    std::chrono::milliseconds time_limit;
    
    static const int piece_values[7];
    static const int pst_pawn[64];
    static const int pst_knight[64];
    static const int pst_bishop[64];
    static const int pst_rook[64];
    static const int pst_queen[64];
    static const int pst_king[64];
    static const int pst_pawn_endgame[64];
    static const int pst_king_endgame[64];
    
public:
    ChessEngine() : board(constants::STARTPOS), stop_search(false), nodes_searched(0), time_limit(5000) {}
    
    void new_game() {
        board.setFen(constants::STARTPOS);
        tt.clear();
    }
    
    void set_position(const std::string& fen) {
        try {
            board.setFen(fen);
        } catch (const std::exception& e) {
            std::cerr << "Error setting position: " << e.what() << std::endl;
            board.setFen(constants::STARTPOS);
        }
    }
    
    void make_move(const std::string& move_str) {
        try {
            Move move = uci::uciToMove(board, move_str);
            if (move != Move::NO_MOVE) {
                Movelist legal_moves;
                movegen::legalmoves(legal_moves, board);
                
                bool is_legal = false;
                for (const auto& legal_move : legal_moves) {
                    if (legal_move == move) {
                        is_legal = true;
                        break;
                    }
                }
                
                if (is_legal) {
                    board.makeMove(move);
                } else {
                    std::cerr << "Illegal move attempted: " << move_str << std::endl;
                }
            } else {
                std::cerr << "Invalid move format: " << move_str << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error making move: " << e.what() << std::endl;
        }
    }
    
    bool is_move_legal(const Move& move) const {
        Movelist legal_moves;
        movegen::legalmoves(legal_moves, board);
        
        for (const auto& legal_move : legal_moves) {
            if (legal_move == move) {
                return true;
            }
        }
        return false;
    }
    
    Move get_first_legal_move() const {
        Movelist legal_moves;
        movegen::legalmoves(legal_moves, board);
        
        if (!legal_moves.empty()) {
            return legal_moves[0];
        }
        return Move::NO_MOVE;
    }
    
    int evaluate() const {
        if (board.isGameOver().first != GameResultReason::NONE) {
            auto result = board.isGameOver();
            if (result.first == GameResultReason::CHECKMATE) {
                return board.sideToMove() == Color::WHITE ? -MATE_VALUE : MATE_VALUE;
            }
            return DRAW_VALUE;
        }
        
        int score = 0;
        Color stm = board.sideToMove();
        
        int piece_count = count_pieces();
        bool is_endgame = piece_count <= 6;
        
        for (int sq = 0; sq < 64; ++sq) {
            Square square(sq);
            Piece piece = board.at(square);
            
            if (piece == Piece::NONE) continue;
            
            Color piece_color = piece.color();
            PieceType piece_type = piece.type();
            
            int piece_value = piece_values[static_cast<int>(piece_type.internal())];
            int positional_value = 0;
            
            int sq_index = piece_color == Color::WHITE ? sq : 63 - sq;
            
            switch (piece_type.internal()) {
                case PieceType::PAWN:
                    if (is_endgame) {
                        positional_value = pst_pawn_endgame[sq_index];
                    } else {
                        positional_value = pst_pawn[sq_index];
                    }
                    break;
                case PieceType::KNIGHT:
                    positional_value = pst_knight[sq_index];
                    break;
                case PieceType::BISHOP:
                    positional_value = pst_bishop[sq_index];
                    break;
                case PieceType::ROOK:
                    positional_value = pst_rook[sq_index];
                    break;
                case PieceType::QUEEN:
                    positional_value = pst_queen[sq_index];
                    break;
                case PieceType::KING:
                    if (is_endgame) {
                        positional_value = pst_king_endgame[sq_index];
                    } else {
                        positional_value = pst_king[sq_index];
                    }
                    break;
                default:
                    break;
            }
            
            int total_value = piece_value + positional_value;
            
            if (piece_color == Color::WHITE) {
                score += total_value;
            } else {
                score -= total_value;
            }
        }
        
        Bitboard white_pawns = board.pieces(PieceType::PAWN, Color::WHITE);
        Bitboard black_pawns = board.pieces(PieceType::PAWN, Color::BLACK);
        
        score += white_pawns.count() * 10 - black_pawns.count() * 10;
        
        int white_mobility = calculate_mobility(Color::WHITE);
        int black_mobility = calculate_mobility(Color::BLACK);
        score += (white_mobility - black_mobility) * 5;
        
        if (is_endgame) {
            Square white_king_sq = board.kingSq(Color::WHITE);
            Square black_king_sq = board.kingSq(Color::BLACK);
            if (stm == Color::WHITE) {
                score += calculate_king_distance_evaluation(white_king_sq, black_king_sq);
            } else {
                score += calculate_king_distance_evaluation(black_king_sq, white_king_sq);
            }
        }
        
        if (board.inCheck()) {
            score += board.sideToMove() == Color::WHITE ? -20 : 20;
        }
        
        return stm == Color::WHITE ? score : -score;
    }
    
    void order_moves(Movelist& moves, Move tt_move) {
        for (auto& move : moves) {
            int score = 0;
            
            if (move == tt_move) {
                score = 10000;
            } else if (board.isCapture(move)) {
                Piece captured = board.at(move.to());
                Piece moving = board.at(move.from());
                
                score = piece_values[static_cast<int>(captured.type().internal())] - 
                       piece_values[static_cast<int>(moving.type().internal())] + 1000;
            } else if (move.typeOf() == Move::PROMOTION) {
                score = piece_values[static_cast<int>(move.promotionType().internal())] + 500;
            }
            
            Board temp_board = board;
            temp_board.makeMove(move);
            if (temp_board.inCheck()) {
                score += 100;
            }
            
            move.setScore(score);
        }
        
        std::sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
            return a.score() > b.score();
        });
    }
    
    int quiescence(int alpha, int beta, int depth = 0) {
        if (depth > 10) return evaluate();
        
        nodes_searched++;
        if (nodes_searched % 1024 == 0) {
            auto now = std::chrono::steady_clock::now();
            if (now - search_start > time_limit) {
                stop_search = true;
                return alpha;
            }
        }
        
        int stand_pat = evaluate();
        
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
        
        Movelist moves;
        movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, board);
        
        order_moves(moves, Move::NO_MOVE);
        
        for (const auto& move : moves) {
            board.makeMove(move);
            int score = -quiescence(-beta, -alpha, depth + 1);
            board.unmakeMove(move);
            
            if (stop_search) return alpha;
            
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        
        return alpha;
    }
    
    int negamax(int depth, int alpha, int beta, bool null_move_allowed = true) {
        if (stop_search) return alpha;
        
        if (depth <= 0) {
            return quiescence(alpha, beta);
        }
        
        nodes_searched++;
        if (nodes_searched % 1024 == 0) {
            auto now = std::chrono::steady_clock::now();
            if (now - search_start > time_limit) {
                stop_search = true;
                return alpha;
            }
        }
        
        std::uint64_t key = board.hash();
        TTEntry* tt_entry = tt.probe(key);
        Move tt_move = Move::NO_MOVE;
        
        if (tt_entry && tt_entry->depth >= depth) {
            if (tt_entry->flag == TT_EXACT) {
                return tt_entry->score;
            } else if (tt_entry->flag == TT_ALPHA && tt_entry->score <= alpha) {
                return alpha;
            } else if (tt_entry->flag == TT_BETA && tt_entry->score >= beta) {
                return beta;
            }
        }
        
        if (tt_entry) {
            tt_move = tt_entry->best_move;
        }
        
        if (board.isHalfMoveDraw() || board.isRepetition()) {
            return DRAW_VALUE;
        }
        
        auto game_result = board.isGameOver();
        if (game_result.first != GameResultReason::NONE) {
            if (game_result.first == GameResultReason::CHECKMATE) {
                return -MATE_VALUE + nodes_searched;
            }
            return DRAW_VALUE;
        }
        
        if (null_move_allowed && depth >= 3 && !board.inCheck() && 
            board.hasNonPawnMaterial(board.sideToMove())) {
            board.makeNullMove();
            int null_score = -negamax(depth - 1 - 2, -beta, -beta + 1, false);
            board.unmakeNullMove();
            
            if (null_score >= beta) {
                return beta;
            }
        }
        
        Movelist moves;
        movegen::legalmoves(moves, board);
        
        if (moves.empty()) {
            if (board.inCheck()) {
                return -MATE_VALUE + nodes_searched;
            }
            return DRAW_VALUE;
        }
        
        order_moves(moves, tt_move);
        
        int best_score = -INF;
        Move best_move = Move::NO_MOVE;
        TTFlag flag = TT_ALPHA;
        
        for (int i = 0; i < moves.size(); ++i) {
            const Move& move = moves[i];
            
            board.makeMove(move);
            
            int score;
            if (i == 0) {
                score = -negamax(depth - 1, -beta, -alpha, true);
            } else {
                score = -negamax(depth - 1, -alpha - 1, -alpha, true);
                if (score > alpha && score < beta) {
                    score = -negamax(depth - 1, -beta, -alpha, true);
                }
            }
            
            board.unmakeMove(move);
            
            if (stop_search) return alpha;
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
            
            if (score >= beta) {
                tt.store(key, best_move, depth, beta, TT_BETA);
                return beta;
            }
            
            if (score > alpha) {
                alpha = score;
                flag = TT_EXACT;
            }
        }
        
        tt.store(key, best_move, depth, best_score, flag);
        return best_score;
    }
    
    Move search(int max_depth = 10) {
        stop_search = false;
        nodes_searched = 0;
        search_start = std::chrono::steady_clock::now();
        
        Move best_move = Move::NO_MOVE;
        
        for (int depth = 1; depth <= max_depth && !stop_search; ++depth) {
            int score = negamax(depth, -INF, INF);
            
            if (!stop_search) {
                TTEntry* entry = tt.probe(board.hash());
                if (entry && entry->best_move != Move::NO_MOVE) {
                    best_move = entry->best_move;
                }
                
                auto elapsed = std::chrono::steady_clock::now() - search_start;
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
                
                std::cout << "info depth " << depth 
                         << " score cp " << score
                         << " nodes " << nodes_searched
                         << " time " << ms
                         << " pv " << uci::moveToUci(best_move) << std::endl;
            }
        }
        
        if (best_move == Move::NO_MOVE || !is_move_legal(best_move)) {
            best_move = get_first_legal_move();
        }
        
        return best_move;
    }
    
    void set_time_limit(int ms) {
        time_limit = std::chrono::milliseconds(ms);
    }
    
    void stop() {
        stop_search = true;
    }
    
    std::string get_fen() const {
        return board.getFen();
    }
    
    int count_pieces() const {
        int count = 0;
        for (int sq = 0; sq < 64; ++sq) {
            Square square(sq);
            Piece piece = board.at(square);
            if (piece != Piece::NONE && piece.type() != PieceType::PAWN && piece.type() != PieceType::KING) {
                count++;
            }
        }
        return count;
    }
    
    int calculate_mobility(Color color) const {
        Movelist moves;
        movegen::legalmoves(moves, board);
        
        int mobility = 0;
        for (const auto& move : moves) {
            Piece piece = board.at(move.from());
            if (piece.color() == color) {
                mobility++;
            }
        }
        return mobility;
    }
    
    int calculate_king_distance_evaluation(Square friendly_king_sq, Square opponent_king_sq) const {
        int friendly_file = friendly_king_sq.file();
        int friendly_rank = friendly_king_sq.rank();
        int opponent_file = opponent_king_sq.file();
        int opponent_rank = opponent_king_sq.rank();

        int opponentKingDstToCentreFile = std::max(3 - opponent_file, opponent_file - 4);
        int opponentKingDstToCentreRank = std::max(3 - opponent_rank, opponent_rank - 4);
        int opponentKingDstFromCentre = opponentKingDstToCentreFile + opponentKingDstToCentreRank;

        int dstBetweenKingsFile = std::abs(friendly_file - opponent_file);
        int dstBetweenKingsRank = std::abs(friendly_rank - opponent_rank);
        int dstBetweenKings = dstBetweenKingsFile + dstBetweenKingsRank;

        int evaluation = 0;
        evaluation += opponentKingDstFromCentre;
        evaluation += 14 - dstBetweenKings;
        return evaluation * 10;
    }
};

const int ChessEngine::piece_values[7] = {100, 320, 330, 500, 900, 20000, 0};

const int ChessEngine::pst_pawn[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

const int ChessEngine::pst_knight[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

const int ChessEngine::pst_bishop[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};

const int ChessEngine::pst_rook[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
};

const int ChessEngine::pst_queen[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};

const int ChessEngine::pst_king[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};

const int ChessEngine::pst_pawn_endgame[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    60, 60, 60, 60, 60, 60, 60, 60,
    40, 40, 40, 40, 40, 40, 40, 40,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0
};

const int ChessEngine::pst_king_endgame[64] = {
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -50,-40,-30,-20,-20,-30,-40,-50
};

class UCIInterface {
private:
    ChessEngine engine;
    
public:
    void run() {
        std::string line;
        
        while (std::getline(std::cin, line)) {
            try {
                std::istringstream iss(line);
                std::string command;
                iss >> command;
                
                if (command == "uci") {
                    std::cout << "id name ChessEngine" << std::endl;
                    std::cout << "id author Assistant" << std::endl;
                    std::cout << "uciok" << std::endl;
                }
                else if (command == "isready") {
                    std::cout << "readyok" << std::endl;
                }
                else if (command == "ucinewgame") {
                    engine.new_game();
                }
                else if (command == "position") {
                    std::string type;
                    iss >> type;
                    
                    if (type == "startpos") {
                        engine.set_position(constants::STARTPOS);
                        
                        std::string moves_token;
                        if (iss >> moves_token && moves_token == "moves") {
                            std::string move;
                            while (iss >> move) {
                                engine.make_move(move);
                            }
                        }
                    }
                    else if (type == "fen") {
                        std::string fen;
                        std::string token;
                        for (int i = 0; i < 6 && iss >> token; ++i) {
                            if (i > 0) fen += " ";
                            fen += token;
                        }
                        engine.set_position(fen);
                        
                        std::string moves_token;
                        if (iss >> moves_token && moves_token == "moves") {
                            std::string move;
                            while (iss >> move) {
                                engine.make_move(move);
                            }
                        }
                    }
                }
                else if (command == "go") {
                    int depth = 10;
                    int movetime = 5000;
                    
                    std::string param;
                    while (iss >> param) {
                        if (param == "depth") {
                            iss >> depth;
                        }
                        else if (param == "movetime") {
                            iss >> movetime;
                            engine.set_time_limit(movetime);
                        }
                        else if (param == "wtime" || param == "btime") {
                            int time;
                            iss >> time;
                            engine.set_time_limit(time / 20);
                        }
                    }
                    
                    Move best_move = engine.search(depth);
                    
                    if (best_move == Move::NO_MOVE) {
                        std::cerr << "No legal move found!" << std::endl;
                        best_move = engine.get_first_legal_move();
                    }
                    
                    if (best_move != Move::NO_MOVE) {
                        std::cout << "bestmove " << uci::moveToUci(best_move) << std::endl;
                    } else {
                        std::cerr << "No legal moves available!" << std::endl;
                        std::cout << "bestmove 0000" << std::endl;
                    }
                }
                else if (command == "stop") {
                    engine.stop();
                }
                else if (command == "quit") {
                    break;
                }
            } catch (const std::exception& e) {
                std::cerr << "UCI Error: " << e.what() << std::endl;
            }
        }
    }
};

int main() {
    attacks::initAttacks();
    
    UCIInterface uci;
    uci.run();
    
    return 0;
} 