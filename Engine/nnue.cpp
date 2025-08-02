#include "nnue.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

namespace NNUE {

NNUEEvaluator g_nnue_evaluator;

bool NNUEWeights::load_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open NNUE weights file: " << filename << std::endl;
        return false;
    }
    
    try {
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
        
        if (hidden_size <= 0 || hidden_size > 4096) {
            std::cerr << "Invalid hidden size: " << hidden_size << std::endl;
            return false;
        }
        
        input_weights.resize(INPUT_SIZE * hidden_size);
        input_bias.resize(hidden_size);
        output_weights.resize(2 * hidden_size);
        
        file.read(reinterpret_cast<char*>(input_weights.data()), 
                 INPUT_SIZE * hidden_size * sizeof(std::int16_t));
        
        file.read(reinterpret_cast<char*>(input_bias.data()), 
                 hidden_size * sizeof(std::int16_t));
        
        file.read(reinterpret_cast<char*>(output_weights.data()), 
                 2 * hidden_size * sizeof(std::int16_t));
        
        file.read(reinterpret_cast<char*>(&output_bias), sizeof(output_bias));
        
        if (file.fail()) {
            std::cerr << "Error reading NNUE weights file" << std::endl;
            return false;
        }
        
        file.close();
        std::cout << "Loaded NNUE weights from " << filename << std::endl;
        print_info();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception loading NNUE weights: " << e.what() << std::endl;
        return false;
    }
}

void NNUEWeights::print_info() const {
    std::cout << "NNUE Network Info:" << std::endl;
    std::cout << "  Hidden size: " << hidden_size << std::endl;
    std::cout << "  Input weights: " << input_weights.size() << std::endl;
    std::cout << "  Input bias: " << input_bias.size() << std::endl;
    std::cout << "  Output weights: " << output_weights.size() << std::endl;
    std::cout << "  Output bias: " << output_bias << std::endl;
}

NNUEEvaluator::NNUEEvaluator() : current_ply(0), initialized(false) {
    accumulator_stack.reserve(256);
    feature_changes.reserve(256);
}

bool NNUEEvaluator::initialize(const std::string& weights_file) {
    if (!weights.load_from_file(weights_file)) {
        return false;
    }
    
    accumulator_stack.clear();
    feature_changes.clear();
    current_ply = 0;
    
    accumulator_stack.emplace_back(weights.hidden_size);
    feature_changes.emplace_back();
    
    initialized = true;
    return true;
}

void NNUEEvaluator::extract_features(const chess::Board& board, std::vector<int> features[2]) {
    features[0].clear();
    features[1].clear();
    
    for (int sq = 0; sq < 64; ++sq) {
        chess::Square square(sq);
        chess::Piece piece = board.at(square);
        
        if (piece == chess::Piece::NONE) continue;
        
        int piece_type = piece_to_index(piece.type());
        if (piece_type < 0) continue;
        
        int color = piece.color() == chess::Color::WHITE ? 0 : 1;
        
        int white_feature = calculate_feature_index(sq, piece_type, color);
        features[0].push_back(white_feature);
        
        int black_square = sq ^ 56;
        int black_color = 1 - color;
        int black_feature = calculate_feature_index(black_square, piece_type, black_color);
        features[1].push_back(black_feature);
    }
}

void NNUEEvaluator::compute_feature_delta(const chess::Board& before, const chess::Board& after, FeatureChange& change) {
    change.clear();
    
    std::vector<int> before_features[2], after_features[2];
    extract_features(before, before_features);
    extract_features(after, after_features);
    
    for (int perspective = 0; perspective < 2; ++perspective) {
        for (int feature : before_features[perspective]) {
            if (std::find(after_features[perspective].begin(), after_features[perspective].end(), feature) 
                == after_features[perspective].end()) {
                change.removed_features[perspective].push_back(feature);
            }
        }
        
        for (int feature : after_features[perspective]) {
            if (std::find(before_features[perspective].begin(), before_features[perspective].end(), feature) 
                == before_features[perspective].end()) {
                change.added_features[perspective].push_back(feature);
            }
        }
    }
}

void NNUEEvaluator::refresh_accumulator(const chess::Board& board, AccumulatorState& acc_state) {
    acc_state.clear();
    
    std::copy(weights.input_bias.begin(), weights.input_bias.end(), acc_state.white.values.begin());
    std::copy(weights.input_bias.begin(), weights.input_bias.end(), acc_state.black.values.begin());
    
    std::vector<int> features[2];
    extract_features(board, features);
    
    for (int feature : features[0]) {
        add_feature_simd(acc_state.white.values.data(), 
                        &weights.input_weights[feature * weights.hidden_size]);
    }
    
    for (int feature : features[1]) {
        add_feature_simd(acc_state.black.values.data(), 
                        &weights.input_weights[feature * weights.hidden_size]);
    }
    
    acc_state.white.needs_refresh = false;
    acc_state.black.needs_refresh = false;
}

void NNUEEvaluator::update_accumulator_incremental(const FeatureChange& change, AccumulatorState& acc_state) {
    if (change.is_null_move) {
        return;
    }
    
    for (int feature : change.removed_features[0]) {
        sub_feature_simd(acc_state.white.values.data(), 
                        &weights.input_weights[feature * weights.hidden_size]);
    }
    for (int feature : change.added_features[0]) {
        add_feature_simd(acc_state.white.values.data(), 
                        &weights.input_weights[feature * weights.hidden_size]);
    }
    
    for (int feature : change.removed_features[1]) {
        sub_feature_simd(acc_state.black.values.data(), 
                        &weights.input_weights[feature * weights.hidden_size]);
    }
    for (int feature : change.added_features[1]) {
        add_feature_simd(acc_state.black.values.data(), 
                        &weights.input_weights[feature * weights.hidden_size]);
    }
    
    acc_state.white.needs_refresh = false;
    acc_state.black.needs_refresh = false;
}

int NNUEEvaluator::forward_pass(const AccumulatorState& acc_state, chess::Color side_to_move) {
    const std::int16_t* stm_acc = (side_to_move == chess::Color::WHITE) ? 
                                  acc_state.white.values.data() : acc_state.black.values.data();
    const std::int16_t* nstm_acc = (side_to_move == chess::Color::WHITE) ? 
                                   acc_state.black.values.data() : acc_state.white.values.data();
    
    std::int32_t output = 0;
    
    output += screlu_simd(stm_acc, weights.output_weights.data(), weights.hidden_size);
    output += screlu_simd(nstm_acc, &weights.output_weights[weights.hidden_size], weights.hidden_size);
    output += weights.output_bias;
    
    output = (output * SCALE) / (QA * QB);
    
    return static_cast<int>(output);
}

void NNUEEvaluator::add_feature_simd(std::int16_t* accumulator, const std::int16_t* weights) {
    constexpr int SIMD_WIDTH = 16;
    int i = 0;
    
    for (; i + SIMD_WIDTH <= this->weights.hidden_size; i += SIMD_WIDTH) {
        __m256i acc_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&accumulator[i]));
        __m256i weight_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&weights[i]));
        __m256i result = _mm256_add_epi16(acc_vec, weight_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&accumulator[i]), result);
    }
    
    for (; i < this->weights.hidden_size; ++i) {
        accumulator[i] += weights[i];
    }
}

void NNUEEvaluator::sub_feature_simd(std::int16_t* accumulator, const std::int16_t* weights) {
    constexpr int SIMD_WIDTH = 16;
    int i = 0;
    
    for (; i + SIMD_WIDTH <= this->weights.hidden_size; i += SIMD_WIDTH) {
        __m256i acc_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&accumulator[i]));
        __m256i weight_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&weights[i]));
        __m256i result = _mm256_sub_epi16(acc_vec, weight_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&accumulator[i]), result);
    }
    
    for (; i < this->weights.hidden_size; ++i) {
        accumulator[i] -= weights[i];
    }
}

std::int32_t NNUEEvaluator::screlu_simd(const std::int16_t* input, const std::int16_t* weights, int size) {
    constexpr int SIMD_WIDTH = 16;
    __m256i sum_vec = _mm256_setzero_si256();
    __m256i zero_vec = _mm256_setzero_si256();
    __m256i qa_vec = _mm256_set1_epi16(QA);
    
    int i = 0;
    
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m256i input_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input[i]));
        
        __m256i clamped = _mm256_max_epi16(input_vec, zero_vec);
        clamped = _mm256_min_epi16(clamped, qa_vec);
        
        __m256i weight_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&weights[i]));
        
        __m256i product = _mm256_mullo_epi16(clamped, weight_vec);
        __m256i squared = _mm256_mullo_epi16(product, clamped);
        
        __m256i squared_lo = _mm256_unpacklo_epi16(squared, _mm256_setzero_si256());
        __m256i squared_hi = _mm256_unpackhi_epi16(squared, _mm256_setzero_si256());
        
        squared_lo = _mm256_srai_epi32(_mm256_slli_epi32(squared_lo, 16), 16);
        squared_hi = _mm256_srai_epi32(_mm256_slli_epi32(squared_hi, 16), 16);
        
        sum_vec = _mm256_add_epi32(sum_vec, squared_lo);
        sum_vec = _mm256_add_epi32(sum_vec, squared_hi);
    }
    
    std::int32_t result[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(result), sum_vec);
    std::int32_t total = result[0] + result[1] + result[2] + result[3] + 
                         result[4] + result[5] + result[6] + result[7];
    
    for (; i < size; ++i) {
        std::int16_t clamped = std::max(std::int16_t(0), std::min(input[i], std::int16_t(QA)));
        std::int32_t product = static_cast<std::int32_t>(clamped) * weights[i];
        total += product * clamped;
    }
    
    return total / QA;
}

void NNUEEvaluator::set_position(const chess::Board& board) {
    if (!initialized) return;
    
    current_ply = 0;
    accumulator_stack.resize(1);
    feature_changes.resize(1);
    
    refresh_accumulator(board, accumulator_stack[0]);
}

void NNUEEvaluator::make_move(const chess::Move& move, const chess::Board& before, const chess::Board& after) {
    if (!initialized) return;
    
    ++current_ply;
    if (current_ply >= accumulator_stack.size()) {
        accumulator_stack.emplace_back(weights.hidden_size);
        feature_changes.emplace_back();
    }
    
    accumulator_stack[current_ply] = accumulator_stack[current_ply - 1];
    
    compute_feature_delta(before, after, feature_changes[current_ply]);
    
    update_accumulator_incremental(feature_changes[current_ply], accumulator_stack[current_ply]);
}

void NNUEEvaluator::unmake_move() {
    if (!initialized || current_ply <= 0) return;
    --current_ply;
}

void NNUEEvaluator::make_null_move() {
    if (!initialized) return;
    
    ++current_ply;
    if (current_ply >= accumulator_stack.size()) {
        accumulator_stack.emplace_back(weights.hidden_size);
        feature_changes.emplace_back();
    }
    
    accumulator_stack[current_ply] = accumulator_stack[current_ply - 1];
    feature_changes[current_ply].is_null_move = true;
}

void NNUEEvaluator::unmake_null_move() {
    unmake_move();
}

int NNUEEvaluator::evaluate(const chess::Board& board, chess::Color side_to_move) {
    if (!initialized) {
        return 0;
    }
    
    if (current_ply >= accumulator_stack.size()) {
        set_position(board);
    }
    
    AccumulatorState& acc_state = accumulator_stack[current_ply];
    
    if (acc_state.white.needs_refresh || acc_state.black.needs_refresh) {
        refresh_accumulator(board, acc_state);
    }
    
    return forward_pass(acc_state, side_to_move);
}

void NNUEEvaluator::clear_cache() {
    if (!initialized) return;
    
    current_ply = 0;
    accumulator_stack.resize(1);
    feature_changes.resize(1);
    accumulator_stack[0].clear();
}

std::string NNUEEvaluator::get_info() const {
    std::ostringstream oss;
    oss << "NNUE Evaluator Status:\n";
    oss << "  Initialized: " << (initialized ? "Yes" : "No") << "\n";
    if (initialized) {
        oss << "  Hidden size: " << weights.hidden_size << "\n";
        oss << "  Current ply: " << current_ply << "\n";
        oss << "  Stack size: " << accumulator_stack.size() << "\n";
    }
    return oss.str();
}

bool initialize_nnue(const std::string& weights_file) {
    return g_nnue_evaluator.initialize(weights_file);
}

int evaluate_nnue(const chess::Board& board, chess::Color side_to_move) {
    return g_nnue_evaluator.evaluate(board, side_to_move);
}

void set_nnue_position(const chess::Board& board) {
    g_nnue_evaluator.set_position(board);
}

} // namespace NNUE 