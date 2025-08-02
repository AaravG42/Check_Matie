#pragma once

#include "chess.hpp"
#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <fstream>
#include <iostream>
#include <cstring>

namespace NNUE {

constexpr int INPUT_SIZE = 768;
constexpr int SCALE = 400;
constexpr int QA = 255;
constexpr int QB = 64;

class NNUEEvaluator;

inline int calculate_feature_index(int square, int piece_type, int color) {
    return color * 64 * 6 + piece_type * 64 + square;
}

inline int piece_to_index(chess::PieceType piece_type) {
    switch (piece_type.internal()) {
        case chess::PieceType::PAWN:   return 0;
        case chess::PieceType::KNIGHT: return 1;
        case chess::PieceType::BISHOP: return 2;
        case chess::PieceType::ROOK:   return 3;
        case chess::PieceType::QUEEN:  return 4;
        case chess::PieceType::KING:   return 5;
        default: return -1;
    }
}

struct Accumulator {
    std::vector<std::int16_t> values;
    bool needs_refresh;
    
    Accumulator(int size) : values(size, 0), needs_refresh(true) {}
    
    void clear() {
        std::fill(values.begin(), values.end(), 0);
        needs_refresh = true;
    }
};

struct NNUEWeights {
    int hidden_size;
    std::vector<std::int16_t> input_weights;
    std::vector<std::int16_t> input_bias;
    std::vector<std::int16_t> output_weights;
    std::int32_t output_bias;
    
    NNUEWeights() : hidden_size(0), output_bias(0) {}
    
    bool load_from_file(const std::string& filename);
    void print_info() const;
};

struct AccumulatorState {
    Accumulator white;
    Accumulator black;
    
    AccumulatorState() : white(256), black(256) {}
    AccumulatorState(int hidden_size) : white(hidden_size), black(hidden_size) {}
    
    void clear() {
        white.clear();
        black.clear();
    }
};

class NNUEEvaluator {
private:
    NNUEWeights weights;
    std::vector<AccumulatorState> accumulator_stack;
    int current_ply;
    bool initialized;
    
    struct FeatureChange {
        std::vector<int> added_features[2];
        std::vector<int> removed_features[2];
        bool is_null_move;
        
        FeatureChange() : is_null_move(false) {}
        
        void clear() {
            added_features[0].clear();
            added_features[1].clear();
            removed_features[0].clear();
            removed_features[1].clear();
            is_null_move = false;
        }
    };
    
    std::vector<FeatureChange> feature_changes;
    
    void extract_features(const chess::Board& board, std::vector<int> features[2]);
    void compute_feature_delta(const chess::Board& before, const chess::Board& after, FeatureChange& change);
    void refresh_accumulator(const chess::Board& board, AccumulatorState& acc_state);
    void update_accumulator_incremental(const FeatureChange& change, AccumulatorState& acc_state);
    int forward_pass(const AccumulatorState& acc_state, chess::Color side_to_move);
    
    void add_feature_simd(std::int16_t* accumulator, const std::int16_t* weights);
    void sub_feature_simd(std::int16_t* accumulator, const std::int16_t* weights);
    std::int32_t screlu_simd(const std::int16_t* input, const std::int16_t* weights, int size);
    
public:
    NNUEEvaluator();
    ~NNUEEvaluator() = default;
    
    bool initialize(const std::string& weights_file);
    bool is_initialized() const { return initialized; }
    
    void set_position(const chess::Board& board);
    void make_move(const chess::Move& move, const chess::Board& before, const chess::Board& after);
    void unmake_move();
    void make_null_move();
    void unmake_null_move();
    
    int evaluate(const chess::Board& board, chess::Color side_to_move);
    
    void clear_cache();
    std::string get_info() const;
};

extern NNUEEvaluator g_nnue_evaluator;

bool initialize_nnue(const std::string& weights_file);
int evaluate_nnue(const chess::Board& board, chess::Color side_to_move);
void set_nnue_position(const chess::Board& board);

} // namespace NNUE 