#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include "chess.hpp"

using namespace chess;

// Only store evaluation values in the transposition table
std::unordered_map<std::string, int> board_positions_val_dict;
std::vector<std::vector<Move>> visited_histories_list;

class ChessHistory {
private:
    Board board;
    std::vector<Move> history;
    int max_depth;

public:
    ChessHistory(const std::string& fen, int depth) : board(fen), max_depth(depth) {}

    std::string get_board_str() const {
        return board.getFen();
    }

    bool is_terminal() const {
        return board.isGameOver().first != GameResultReason::NONE || history.size() >= max_depth;
    }

    int get_value() const {
        auto [reason, result] = board.isGameOver();
        if (reason == GameResultReason::CHECKMATE) {
            return board.sideToMove() == Color::WHITE ? -1 : 1;
        }
        return 0; // Draw or max depth reached
    }

    std::vector<Move> get_valid_moves() const {
        Movelist moves;
        movegen::legalmoves(moves, board);
        return std::vector<Move>(moves.begin(), moves.end());
    }

    ChessHistory make_move(const Move& move) const {
        ChessHistory new_history = *this;
        new_history.board.makeMove(move);
        new_history.history.push_back(move);
        return new_history;
    }

    const std::vector<Move>& get_history() const {
        return history;
    }
};

std::pair<int, std::vector<Move>> alpha_beta_pruning(const ChessHistory& history, int alpha, int beta, bool max_player, int depth) {
    // Check if position was already evaluated
    auto it = board_positions_val_dict.find(history.get_board_str());
    if (it != board_positions_val_dict.end()) {
        return {it->second, {}}; // Return empty move list for non-winning positions
    }

    visited_histories_list.push_back(history.get_history());

    if (history.is_terminal() || depth == 0) {
        int value = history.get_value();
        board_positions_val_dict[history.get_board_str()] = value;
        return {value, {}}; // Return empty move list for non-winning positions
    }

    if (max_player) {
        int best_value = std::numeric_limits<int>::min();
        std::vector<Move> best_moves;
        
        for (const auto& move : history.get_valid_moves()) {
            ChessHistory new_history = history.make_move(move);
            auto [value, moves] = alpha_beta_pruning(new_history, alpha, beta, false, depth - 1);
            
            if (value > best_value) {
                best_value = value;
                // Only store moves if we found a winning solution
                if (value == 1) {
                    best_moves = moves;
                    best_moves.insert(best_moves.begin(), move);
                }
            }
            
            alpha = std::max(alpha, best_value);
            if (alpha >= beta) {
                break;
            }
        }
        
        board_positions_val_dict[history.get_board_str()] = best_value;
        return {best_value, best_moves};
    } else {
        int best_value = std::numeric_limits<int>::max();
        std::vector<Move> best_moves;
        
        for (const auto& move : history.get_valid_moves()) {
            ChessHistory new_history = history.make_move(move);
            auto [value, moves] = alpha_beta_pruning(new_history, alpha, beta, true, depth - 1);
            
            if (value < best_value) {
                best_value = value;
                // Only store moves if we found a winning solution
                if (value == -1) {
                    best_moves = moves;
                    best_moves.insert(best_moves.begin(), move);
                }
            }
            
            beta = std::min(beta, best_value);
            if (alpha >= beta) {
                break;
            }
        }
        
        board_positions_val_dict[history.get_board_str()] = best_value;
        return {best_value, best_moves};
    }
}

std::pair<int, std::vector<Move>> solve_mate_in_n(const std::string& fen, int depth) {
    ChessHistory history(fen, depth);
    return alpha_beta_pruning(history, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), true, depth);
}

void print_moves(const std::vector<Move>& moves, const Board& board) {
    Board temp_board = board;
    for (const auto& move : moves) {
        std::cout << uci::moveToSan(temp_board, move) << " ";
        temp_board.makeMove(move);
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <fen_string> <depth>" << std::endl;
        return 1;
    }

    std::string fen = argv[1];
    int depth = std::stoi(argv[2]);

    Board initial_board(fen);
    auto [value, winning_moves] = solve_mate_in_n(fen, depth);
    
    if (value == 1 || value == -1) {
        print_moves(winning_moves, initial_board);
    } else {
        std::cout << "No mate found within " << depth << " moves" << std::endl;
    }

    return 0;
}
