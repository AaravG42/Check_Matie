#include "chess.hpp"
#include "nnue.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <climits>
#include <fstream>
#include <vector>
#include <random>
#include <cstring>

#include <endian.h>

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

struct BookEntry {
    std::uint64_t key;
    std::uint16_t move;
    std::uint16_t weight;
    std::uint32_t learn;
    
    BookEntry() : key(0), move(0), weight(0), learn(0) {}
    BookEntry(std::uint64_t k, std::uint16_t m, std::uint16_t w, std::uint32_t l) 
        : key(k), move(m), weight(w), learn(l) {}
};

class PolyglotBook {
private:
    std::vector<BookEntry> entries;
    std::vector<std::uint64_t> polyglot_random;
    bool loaded;
    
    void generatePolyglotRandom() {
        polyglot_random.resize(781);
        
        std::mt19937_64 gen(0x9D39247E33776D41ULL);
        std::uniform_int_distribution<std::uint64_t> dist;
        
        for (size_t i = 0; i < 781; ++i) {
            polyglot_random[i] = dist(gen);
        }
    }
    
    std::uint64_t getPolyglotKey(const Board& board) const {
        std::uint64_t key = 0;
        
        for (int sq = 0; sq < 64; ++sq) {
            Square square(sq);
            Piece piece = board.at(square);
            
            if (piece != Piece::NONE) {
                int piece_index = getPieceIndex(piece);
                if (piece_index >= 0) {
                    key ^= polyglot_random[64 * piece_index + sq];
                }
            }
        }
        
        if (board.castlingRights().has(Color::WHITE, Board::CastlingRights::Side::KING_SIDE)) {
            key ^= polyglot_random[768];
        }
        if (board.castlingRights().has(Color::WHITE, Board::CastlingRights::Side::QUEEN_SIDE)) {
            key ^= polyglot_random[769];
        }
        if (board.castlingRights().has(Color::BLACK, Board::CastlingRights::Side::KING_SIDE)) {
            key ^= polyglot_random[770];
        }
        if (board.castlingRights().has(Color::BLACK, Board::CastlingRights::Side::QUEEN_SIDE)) {
            key ^= polyglot_random[771];
        }
        
        Square ep_sq = board.enpassantSq();
        if (ep_sq != Square::underlying::NO_SQ) {
            key ^= polyglot_random[772 + ep_sq.file()];
        }
        
        if (board.sideToMove() == Color::WHITE) {
            key ^= polyglot_random[780];
        }
        
        return key;
    }
    
    int getPieceIndex(Piece piece) const {
        Color color = piece.color();
        PieceType type = piece.type();
        
        int type_offset = 0;
        switch (type.internal()) {
            case PieceType::PAWN: type_offset = 0; break;
            case PieceType::KNIGHT: type_offset = 2; break;
            case PieceType::BISHOP: type_offset = 4; break;
            case PieceType::ROOK: type_offset = 6; break;
            case PieceType::QUEEN: type_offset = 8; break;
            case PieceType::KING: type_offset = 10; break;
            default: return -1;
        }
        
        return type_offset + (color == Color::WHITE ? 1 : 0);
    }
    
    Move polyglotMoveToMove(const Board& board, std::uint16_t poly_move) const {
        int from_sq = (poly_move >> 6) & 63;
        int to_sq = poly_move & 63;
        int promo = (poly_move >> 12) & 7;
        
        Square from(from_sq);
        Square to(to_sq);
        
        Movelist legal_moves;
        movegen::legalmoves(legal_moves, board);
        
        for (const auto& move : legal_moves) {
            if (move.from() == from && move.to() == to) {
                if (promo == 0 || 
                    (promo == 1 && move.promotionType() == PieceType::KNIGHT) ||
                    (promo == 2 && move.promotionType() == PieceType::BISHOP) ||
                    (promo == 3 && move.promotionType() == PieceType::ROOK) ||
                    (promo == 4 && move.promotionType() == PieceType::QUEEN)) {
                    return move;
                }
            }
        }
        
        return Move::NO_MOVE;
    }
    
public:
    PolyglotBook() : loaded(false) {
        generatePolyglotRandom();
    }
    
    bool loadBook(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open book file: " << filename << std::endl;
            return false;
        }
        
        entries.clear();
        
        char buffer[16];
        while (file.read(buffer, 16)) {
            BookEntry entry;
            
            std::memcpy(&entry.key, buffer, 8);
            std::memcpy(&entry.move, buffer + 8, 2);
            std::memcpy(&entry.weight, buffer + 10, 2);
            std::memcpy(&entry.learn, buffer + 12, 4);
            
            entry.key = be64toh(entry.key);
            entry.move = be16toh(entry.move);
            entry.weight = be16toh(entry.weight);
            entry.learn = be32toh(entry.learn);
            
            entries.push_back(entry);
        }
        
        std::sort(entries.begin(), entries.end(), 
                 [](const BookEntry& a, const BookEntry& b) {
                     return a.key < b.key;
                 });
        
        loaded = !entries.empty();
        std::cout << "Loaded " << entries.size() << " book entries from " << filename << std::endl;
        return loaded;
    }
    
    std::vector<BookEntry> getBookMoves(const Board& board) const {
        if (!loaded) return {};
        
        std::uint64_t position_key = getPolyglotKey(board);
        std::vector<BookEntry> book_moves;
        
        auto range = std::equal_range(entries.begin(), entries.end(),
            BookEntry(position_key, 0, 0, 0),
            [](const BookEntry& a, const BookEntry& b) {
                return a.key < b.key;
            });
        
        for (auto it = range.first; it != range.second; ++it) {
            Move move = polyglotMoveToMove(board, it->move);
            if (move != Move::NO_MOVE) {
                book_moves.push_back(*it);
            }
        }
        
        return book_moves;
    }
    
    Move selectBookMove(const Board& board) const {
        auto book_moves = getBookMoves(board);
        if (book_moves.empty()) return Move::NO_MOVE;
        
        std::uint32_t total_weight = 0;
        for (const auto& entry : book_moves) {
            total_weight += entry.weight;
        }
        
        if (total_weight == 0) return Move::NO_MOVE;
        
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<std::uint32_t> dist(0, total_weight - 1);
        std::uint32_t random_val = dist(gen);
        
        std::uint32_t current_weight = 0;
        for (const auto& entry : book_moves) {
            current_weight += entry.weight;
            if (random_val < current_weight) {
                return polyglotMoveToMove(board, entry.move);
            }
        }
        
        return Move::NO_MOVE;
    }
    
    bool isLoaded() const { return loaded; }
    size_t getEntryCount() const { return entries.size(); }
};

class ChessEngine {
private:
    Board board;
    TranspositionTable tt;
    PolyglotBook opening_book;
    bool stop_search;
    int nodes_searched;
    std::chrono::steady_clock::time_point search_start;
    std::chrono::milliseconds time_limit;
    
    bool use_nnue;
    bool nnue_initialized;
    
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
    ChessEngine() : board(constants::STARTPOS), stop_search(false), nodes_searched(0), 
                   time_limit(5000), use_nnue(false), nnue_initialized(false) {}
    
    bool initialize_nnue_eval(const std::string& weights_file) {
        nnue_initialized = NNUE::initialize_nnue(weights_file);
        if (nnue_initialized) {
            use_nnue = true;
            std::cout << "NNUE evaluation initialized successfully" << std::endl;
        } else {
            use_nnue = false;
            std::cout << "Failed to initialize NNUE evaluation" << std::endl;
        }
        return nnue_initialized;
    }
    
    void set_use_nnue(bool enable) {
        if (enable && !nnue_initialized) {
            std::cerr << "Cannot enable NNUE: not initialized" << std::endl;
            return;
        }
        use_nnue = enable;
    }
    
    void new_game() {
        board.setFen(constants::STARTPOS);
        tt.clear();
        if (nnue_initialized) {
            NNUE::set_nnue_position(board);
        }
    }
    
    void set_position(const std::string& fen) {
        try {
            board.setFen(fen);
            if (nnue_initialized) {
                NNUE::set_nnue_position(board);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error setting position: " << e.what() << std::endl;
            board.setFen(constants::STARTPOS);
            if (nnue_initialized) {
                NNUE::set_nnue_position(board);
            }
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
                    Board before = board;
                    board.makeMove(move);
                    
                    if (nnue_initialized) {
                        NNUE::g_nnue_evaluator.make_move(move, before, board);
                    }
                    
                    tt.clear();
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
        
        if (use_nnue && nnue_initialized) {
            if (should_use_classical_evaluation()) {
                return evaluate_classical();
            } else {
                int nnue_eval = NNUE::evaluate_nnue(board, board.sideToMove());
                int classical_eval = evaluate_classical();
                
                return (nnue_eval * 9 + classical_eval) / 10;
            }
        } else {
            return evaluate_classical();
        }
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
            } else {
                // In endgames, prioritize checking moves to find mates faster
                board.makeMove(move);
                if (board.inCheck()) {
                    score += 200;  // Bonus for giving check
                }
                board.unmakeMove(move);
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
                return -MATE_VALUE - depth;  // Favor shorter mates
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
                return -MATE_VALUE - depth;  // Favor shorter mates
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
        
        if (opening_book.isLoaded()) {
            Move book_move = opening_book.selectBookMove(board);
            if (book_move != Move::NO_MOVE) {
                std::cout << "info string Using opening book move" << std::endl;
                return book_move;
            }
        }
        
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
    
    int evaluate_mating_patterns(Color side_to_move) const {
        int score = 0;
        Color opponent = side_to_move == Color::WHITE ? Color::BLACK : Color::WHITE;
        
        Square opponent_king = board.kingSq(opponent);
        Square my_king = board.kingSq(side_to_move);
        
        // Count material for both sides
        int my_queens = board.pieces(PieceType::QUEEN, side_to_move).count();
        int my_rooks = board.pieces(PieceType::ROOK, side_to_move).count(); 
        int my_bishops = board.pieces(PieceType::BISHOP, side_to_move).count();
        int my_knights = board.pieces(PieceType::KNIGHT, side_to_move).count();
        
        int opp_queens = board.pieces(PieceType::QUEEN, opponent).count();
        int opp_rooks = board.pieces(PieceType::ROOK, opponent).count();
        int opp_bishops = board.pieces(PieceType::BISHOP, opponent).count();
        int opp_knights = board.pieces(PieceType::KNIGHT, opponent).count();
        
        // Basic mate detection patterns
        bool sufficient_material = (my_queens >= 1) || 
                                  (my_rooks >= 2) || 
                                  (my_rooks >= 1 && (my_bishops >= 1 || my_knights >= 1)) ||
                                  (my_rooks >= 1 && my_queens >= 1);
        
        if (sufficient_material && (opp_queens + opp_rooks + opp_bishops + opp_knights == 0)) {
            // We have mating material vs lone king
            int king_file = opponent_king.file();
            int king_rank = opponent_king.rank();
            
            // Reward pushing enemy king to edges
            int edge_distance = std::min({king_file, 7 - king_file, king_rank, 7 - king_rank});
            score += (3 - edge_distance) * 50;
            
            // Reward keeping our king close for support
            int king_distance = std::abs(my_king.file() - opponent_king.file()) + 
                               std::abs(my_king.rank() - opponent_king.rank());
            score += (7 - king_distance) * 20;
            
            // Special bonus for corner squares in basic mates
            if ((king_file == 0 || king_file == 7) && (king_rank == 0 || king_rank == 7)) {
                score += 100;
            }
            
            // Lawnmower mate pattern recognition
            if (my_rooks >= 2 || (my_rooks >= 1 && my_queens >= 1)) {
                // Reward pieces on same rank/file when enemy king is on edge
                if (edge_distance <= 1) {
                    // Find our major pieces by iterating through all squares
                    std::vector<Square> piece_squares;
                    for (int sq = 0; sq < 64; ++sq) {
                        Square square(sq);
                        Piece piece = board.at(square);
                        if (piece.color() == side_to_move && 
                            (piece.type() == PieceType::ROOK || piece.type() == PieceType::QUEEN)) {
                            piece_squares.push_back(square);
                        }
                    }
                    
                    for (size_t i = 0; i < piece_squares.size(); ++i) {
                        for (size_t j = i + 1; j < piece_squares.size(); ++j) {
                            Square sq1 = piece_squares[i];
                            Square sq2 = piece_squares[j];
                            // Same rank or file = good for lawnmower
                            if (sq1.rank() == sq2.rank() || sq1.file() == sq2.file()) {
                                score += 30;
                            }
                        }
                    }
                }
            }
            
            // Massive bonus for actual checkmate detection
            Movelist moves;
            movegen::legalmoves(moves, board);
            if (moves.empty() && board.inCheck()) {
                score += 1000; // This is checkmate!
            }
        }
        
        return score;
    }
    
    bool loadOpeningBook(const std::string& filename) {
        return opening_book.loadBook(filename);
    }
    
    bool isBookLoaded() const {
        return opening_book.isLoaded();
    }
    
    size_t getBookEntryCount() const {
        return opening_book.getEntryCount();
    }
    
    std::string get_engine_info() const {
        std::ostringstream oss;
        oss << "Engine Info:\n";
        oss << "  NNUE Available: " << (nnue_initialized ? "Yes" : "No") << "\n";
        oss << "  NNUE Enabled: " << (use_nnue ? "Yes" : "No") << "\n";
        oss << "  Position: " << board.getFen() << "\n";
        
        if (nnue_initialized) {
            oss << NNUE::g_nnue_evaluator.get_info();
        }
        
        return oss.str();
    }
    
private:
    bool should_use_classical_evaluation() const {
        int move_count = board.fullMoveNumber() * 2 - (board.sideToMove() == Color::WHITE ? 2 : 1);
        if (move_count < 6) {
            return true;
        }
        
        int piece_count = count_pieces();
        if (piece_count <= 4) {
            return true;
        }
        
        int white_material = 0, black_material = 0;
        for (int sq = 0; sq < 64; ++sq) {
            Square square(sq);
            Piece piece = board.at(square);
            
            if (piece == Piece::NONE) continue;
            
            int value = piece_values[static_cast<int>(piece.type().internal())];
            if (piece.color() == Color::WHITE) {
                white_material += value;
            } else {
                black_material += value;
            }
        }
        
        int material_diff = std::abs(white_material - black_material);
        if (material_diff > 1500) {
            return true;
        }
        
        return false;
    }
    
    int evaluate_classical() const {
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
                score -= calculate_king_distance_evaluation(black_king_sq, white_king_sq);
            }
            
            score += evaluate_mating_patterns(stm);
        }
        
        if (board.inCheck()) {
            score += board.sideToMove() == Color::WHITE ? -20 : 20;
        }
        
        return stm == Color::WHITE ? score : -score;
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
public:
    ChessEngine engine;
    UCIInterface() {
        if (engine.loadOpeningBook("Perfect2023.bin")) {
            std::cout << "info string Loaded Perfect2023.bin opening book with " 
                     << engine.getBookEntryCount() << " entries" << std::endl;
        } else {
            std::cout << "info string Perfect2023.bin not found, continuing without opening book" << std::endl;
        }
    }
    
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
                    std::cout << "option name Use NNUE type check default false" << std::endl;
                    std::cout << "option name EvalFile type string default nnue_weights.bin" << std::endl;
                    std::cout << "option name Hash type spin default 16 min 1 max 1024" << std::endl;
                    std::cout << "uciok" << std::endl;
                }
                else if (command == "isready") {
                    std::cout << "readyok" << std::endl;
                }
                else if (command == "setoption") {
                    std::string name_token, name, value_token, value;
                    if (iss >> name_token >> name >> value_token >> value) {
                        if (name == "Use" && iss >> name && name == "NNUE") {
                            engine.set_use_nnue(value == "true");
                            std::cout << "info string NNUE " << (value == "true" ? "enabled" : "disabled") << std::endl;
                        }
                        else if (name == "EvalFile") {
                            if (engine.initialize_nnue_eval(value)) {
                                std::cout << "info string NNUE weights loaded from " << value << std::endl;
                            } else {
                                std::cout << "info string Failed to load NNUE weights from " << value << std::endl;
                            }
                        }
                    }
                }
                else if (command == "loadnnue") {
                    std::string filename;
                    if (iss >> filename) {
                        if (engine.initialize_nnue_eval(filename)) {
                            std::cout << "info string NNUE weights loaded from " << filename << std::endl;
                        } else {
                            std::cout << "info string Failed to load NNUE weights from " << filename << std::endl;
                        }
                    } else {
                        std::cout << "info string Usage: loadnnue <filename>" << std::endl;
                    }
                }
                else if (command == "nnueinfo") {
                    std::cout << "info string " << engine.get_engine_info() << std::endl;
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
                    int depth = 100;
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
                            engine.set_time_limit(time/30);
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
                else if (command == "loadbook") {
                    std::string filename;
                    if (iss >> filename) {
                        if (engine.loadOpeningBook(filename)) {
                            std::cout << "info string Opening book loaded: " << filename 
                                     << " (" << engine.getBookEntryCount() << " entries)" << std::endl;
                        } else {
                            std::cout << "info string Failed to load opening book: " << filename << std::endl;
                        }
                    } else {
                        std::cout << "info string Usage: loadbook <filename>" << std::endl;
                    }
                }
                else if (command == "bookinfo") {
                    if (engine.isBookLoaded()) {
                        std::cout << "info string Opening book loaded with " 
                                 << engine.getBookEntryCount() << " entries" << std::endl;
                    } else {
                        std::cout << "info string No opening book loaded" << std::endl;
                    }
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
    
    std::ifstream config_file("engine_config.txt");
    if (config_file.is_open()) {
        std::string line;
        while (std::getline(config_file, line)) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string option = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                
                if (option == "Use NNUE") {
                    uci.engine.set_use_nnue(value == "true");
                }
                else if (option == "EvalFile") {
                    uci.engine.initialize_nnue_eval(value);
                }
            }
        }
    }
    
    uci.run();
    
    return 0;
} 