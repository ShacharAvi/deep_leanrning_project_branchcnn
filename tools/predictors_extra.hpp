/* Predictors needed by the project spec that MBPlib does not ship.
 *
 * These implement the mbp::Predictor interface so they drop straight into the
 * same simulation loop as MBPlib's own predictors:
 *
 *     if (branch.isConditional()) { p->predict(ip); p->train(branch); }
 *     p->track(branch);
 *
 * Spec baselines 0, 1 and 4.
 */

#ifndef BP_PREDICTORS_EXTRA_HPP_
#define BP_PREDICTORS_EXTRA_HPP_

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>

#include "mbp/core/predictor.hpp"
#include "nlohmann/json.hpp"

namespace bp {

/* Spec baseline 0. Not a serious predictor -- a sanity check, and the bias
 * reference that stops a heavily-skewed branch from looking like a win. */
struct AlwaysTaken : mbp::Predictor {
  bool predict(uint64_t) override { return true; }
  void train(const mbp::Branch&) override {}
  void track(const mbp::Branch&) override {}

  mbp::json metadata_stats() const override {
    return {{"name", "Always Taken"}, {"storage_bits", 0}};
  }
};

/* Spec baseline 1. Useful for detecting strong class imbalance. */
struct AlwaysNotTaken : mbp::Predictor {
  bool predict(uint64_t) override { return false; }
  void train(const mbp::Branch&) override {}
  void track(const mbp::Branch&) override {}

  mbp::json metadata_stats() const override {
    return {{"name", "Always Not Taken"}, {"storage_bits", 0}};
  }
};

/**
 * Spec baseline 4: the original perceptron predictor of Jimenez & Lin,
 * HPCA 2001 (articles/article3.pdf).
 *
 * One perceptron per PC bucket. History bits are bipolar (taken = +1,
 * not taken = -1), so
 *
 *     y = w[0] + sum_i w[i] * x[i]        prediction = (y >= 0)
 *
 * and training on misprediction or low confidence (|y| <= theta) is a single
 * pass of w[i] += t * x[i]. The paper's threshold heuristic is
 * theta = floor(1.93 * h + 14).
 *
 * This is deliberately the *original* formulation, not MBPlib's later
 * HashedPerceptron -- the project reproduces the paper, and the contrast
 * between the two is itself informative. Both are run side by side.
 *
 * @param H           Global history length.
 * @param LOG_N       log2 of the number of perceptrons.
 * @param WEIGHT_BITS Width of each weight, including sign.
 * @param IGNORE_UCD  Track only conditional branches in the global history.
 */
template <int H = 62, int LOG_N = 10, int WEIGHT_BITS = 8,
          bool IGNORE_UCD = false>
struct ClassicPerceptron : mbp::Predictor {
  static constexpr int NUM_PERCEPTRONS = 1 << LOG_N;
  static constexpr int WEIGHTS_PER_PERCEPTRON = H + 1;  // +1 for the bias
  static constexpr int WEIGHT_MAX = (1 << (WEIGHT_BITS - 1)) - 1;
  static constexpr int WEIGHT_MIN = -(1 << (WEIGHT_BITS - 1));
  // Jimenez & Lin's threshold heuristic.
  static constexpr int THETA = static_cast<int>(1.93 * H + 14);
  static constexpr int64_t STORAGE_BITS =
      int64_t{NUM_PERCEPTRONS} * WEIGHTS_PER_PERCEPTRON * WEIGHT_BITS;

  static_assert(WEIGHT_BITS >= 2 && WEIGHT_BITS <= 16);

  std::array<std::array<int16_t, WEIGHTS_PER_PERCEPTRON>, NUM_PERCEPTRONS>
      weights{};
  std::bitset<H> ghist;

  // Cached between predict() and train(); mbp::Predictor::train does not
  // receive the prediction, so the dot product is kept rather than recomputed.
  int32_t output = 0;
  size_t row = 0;
  bool prediction = false;
  bool valid = false;
  uint64_t predicted_ip = 0;

  static constexpr size_t index(uint64_t ip) {
    return (ip >> 2) & (NUM_PERCEPTRONS - 1);
  }

  bool predict(uint64_t ip) override {
    if (valid && predicted_ip == ip) return prediction;

    row = index(ip);
    const auto& w = weights[row];

    int32_t sum = w[0];  // bias, x[0] == 1
    for (int i = 0; i < H; ++i) {
      sum += ghist[i] ? w[i + 1] : -w[i + 1];
    }

    output = sum;
    predicted_ip = ip;
    valid = true;
    return prediction = (sum >= 0);
  }

  void train(const mbp::Branch& b) override {
    predict(b.ip());  // no-op when already cached for this ip

    const bool taken = b.isTaken();
    if (prediction == taken && std::abs(output) > THETA) {
      valid = false;
      return;
    }

    auto& w = weights[row];
    const int t = taken ? 1 : -1;
    w[0] = static_cast<int16_t>(
        std::clamp(w[0] + t, WEIGHT_MIN, WEIGHT_MAX));
    for (int i = 0; i < H; ++i) {
      const int delta = ghist[i] ? t : -t;
      w[i + 1] = static_cast<int16_t>(
          std::clamp(w[i + 1] + delta, WEIGHT_MIN, WEIGHT_MAX));
    }
    valid = false;
  }

  void track(const mbp::Branch& b) override {
    if (IGNORE_UCD && !b.isConditional()) return;
    ghist <<= 1;
    ghist[0] = b.isTaken();
    valid = false;
  }

  mbp::json metadata_stats() const override {
    return {
        {"name", "Jimenez-Lin Perceptron"},
        {"history_length", H},
        {"num_perceptrons", NUM_PERCEPTRONS},
        {"weight_bits", WEIGHT_BITS},
        {"theta", THETA},
        {"track_only_conditional", IGNORE_UCD},
        {"storage_bits", STORAGE_BITS},
    };
  }
};

}  // namespace bp

#endif  // BP_PREDICTORS_EXTRA_HPP_
