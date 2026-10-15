#ifndef MBP_EXAMPLES_HASHED_PERCEPTRON_HPP_
#define MBP_EXAMPLES_HASHED_PERCEPTRON_HPP_

#include <bitset>
#include <vector>

#include "mbp/core/predictor.hpp"
#include "mbp/utils/indexing.hpp"
#include "mbp/utils/saturated_reg.hpp"
#include "nlohmann/json.hpp"

namespace mbp {

/* NOTE(portability): sqrt() and pow() are constexpr builtins under GCC but not
 * under Apple clang's libc++, and H sizes an array so it cannot be demoted to
 * a runtime const. Replaced with a constexpr Newton sqrt and an integer power
 * loop.
 *
 * Verified numerically equivalent to the originals:
 *   - Newton converges to the exactly-rounded double for sqrt(5.0).
 *   - repeated multiplication matches std::pow to 0 ulp for the exponents used.
 *   - HashedPerceptron<4, 16, 12, 18>::H == 259 either way, and sits 0.23 away
 *     from the nearest integer boundary, so the truncation is not marginal.
 */
namespace hashed_perceptron_detail {

constexpr double Sqrt(double v) {
  double x = v;
  for (int i = 0; i < 8; ++i) x = 0.5 * (x + v / x);
  return x;
}

constexpr double IntPow(double base, int exp) {
  double result = 1.0;
  for (int i = 0; i < exp; ++i) result *= base;
  return result;
}

}  // namespace hashed_perceptron_detail

template <int MINH, int NUMT, int T, int MISP_THRESH = 18>
struct HashedPerceptron : Predictor {
  static constexpr double phi =
      (1 + hashed_perceptron_detail::Sqrt(5.0)) / 2;
  // == std::pow(phi, 1 / phi); a fractional exponent needs constexpr exp/log,
  // so the value is inlined at full double precision instead.
  static constexpr double GEOM_RATIO = 1.3463608200348693;
  static constexpr int H =
      2 + MINH * hashed_perceptron_detail::IntPow(
                     GEOM_RATIO, NUMT - 2 > 0 ? NUMT - 2 : 0);
  // Components
  std::array<std::array<i8, (1 << T)>, NUMT> component{};
  std::array<int, NUMT> hlen;
  std::array<BitStreamXorFold, NUMT> ghistFold;
  std::array<char, H> ghist{};
  size_t ghistIdx = 0;  // was read uninitialised in track()
  int theta;
  int misp;
  // Saved Values
  std::array<uint64_t, NUMT> idx{};
  int sum;
  bool prediction;
  bool tracked;
  uint64_t predictedIp;

  HashedPerceptron()
      : theta(10),
        misp(0),
        sum(0),
        prediction(false),
        tracked(true),
        predictedIp(0) {
    hlen[0] = 0;
    ghistFold[0] = BitStreamXorFold(hlen[0], T);
    double power = MINH;
    for (int i = 1; i < NUMT; ++i) {
      hlen[i] = power;
      ghistFold[i] = BitStreamXorFold(hlen[i], T);
      power *= GEOM_RATIO;
    }
  }

  bool predict(uint64_t ip) override {
    if (tracked == false && predictedIp == ip) return prediction;
    tracked = false;
    predictedIp = ip;

    sum = 0;
    uint64_t ipfold = XorFold(ip, T);
    for (size_t i = 0; i < component.size(); ++i) {
      idx[i] = ipfold ^ ghistFold[i].fold();
      sum += static_cast<int>(component[i][idx[i]]);
    }
    return prediction = sum > 0;
  }

  void train(const Branch& b) override {
    // Predict to initialize the variables prediction, sum, etc.
    predict(b.ip());

    if (b.isTaken() == prediction && std::abs(sum) >= theta) return;
    for (size_t i = 0; i < component.size(); ++i) {
      component[i][idx[i]].sumOrSub(b.isTaken());
    }
    if (b.isTaken() != prediction) {
      if (++misp >= MISP_THRESH) {
        misp = 0;
        theta++;
      }
    } else {
      if (--misp <= -MISP_THRESH) {
        misp = 0;
        theta--;
      }
    }
  }

  void track(const Branch& b) override {
    ghistIdx = (ghistIdx == 0 ? ghist.size() : ghistIdx) - 1U;
    ghist[ghistIdx] = b.isTaken();
    for (size_t i = 0; i < NUMT; ++i) {
      unsigned outgoingBit = ghist[(ghistIdx + hlen[i]) % ghist.size()];
      ghistFold[i].shiftInAndOut(1, b.isTaken(), outgoingBit);
    }
    tracked = true;
  }

  json metadata_stats() const override {
    return {
        {"name", "MBPlib Hashed Perceptron"},
        {"min_history_length", MINH},
        {"num_tables", NUMT},
        {"log_tables_size", T},
        {"mispredictions_threshold", MISP_THRESH},
    };
  }
};

}  // namespace mbp

#endif  // MBP_EXAMPLES_HASHED_PERCEPTRON_HPP_
