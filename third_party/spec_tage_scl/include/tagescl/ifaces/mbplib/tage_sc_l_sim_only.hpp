#ifndef TAGESCL_ZIP_HIST_MBPLIB_COMPAT_LAYER_HPP_
#define TAGESCL_ZIP_HIST_MBPLIB_COMPAT_LAYER_HPP_

#include <cassert>
#include <cstdint>
#include <mbp/core/predictor.hpp>
#include <nlohmann/json.hpp>

#include "../../tagescl.hpp"

namespace tagescl {

template <class CONFIG>
struct MbpTageScl : mbp::Predictor {
  using Impl = Tage_SC_L<CONFIG>;
  enum State {
    kNone,
    kPredicted,
    kCommited,
  };

  Impl impl;
  std::uint64_t currentIp;
  std::uint32_t branchId;
  State state;

  MbpTageScl(std::size_t maxInflightBranches)
      : impl(maxInflightBranches), currentIp(0), branchId(0), state(kNone){};

  bool predict(uint64_t ip) override {
    assert(state == kNone or state == kCommited);
    branchId = impl.get_new_branch_id();
    currentIp = ip;
    state = kPredicted;
    return impl.get_prediction(branchId, ip);
  }

  /* NOTE(local): internal state of the prediction just made.
   *
   * Only valid between predict() and the following train()/track(). Everything
   * it exposes is computed at prediction time, so it is available at fetch in a
   * real pipeline -- reading it is not a leak. Used to classify why a branch is
   * hard (project spec section 2.3), not as a model input.
   */
  const Tage_SC_L_Prediction_Info<CONFIG>& lastPredictionInfo() {
    assert(state == kPredicted);
    return impl.prediction_info(branchId);
  }

  void train(const mbp::Branch& b) override {
    assert(state == kPredicted and b.ip() == currentIp);
    Branch_Type type;
    type.is_conditional = b.isConditional();
    type.is_indirect = b.isIndirect();
    impl.update_speculative_state(branchId, b.ip(), type, b.isTaken(),
                                  b.target());
    impl.commit_state(branchId, b.ip(), type, b.isTaken());
    impl.commit_state_at_retire(branchId, b.ip(), type, b.isTaken(),
                                b.target());
    state = kCommited;
  }

  void track(const mbp::Branch& b) override {
    if (state == kNone) {
      predict(b.ip());
    }
    if (state == kPredicted) {
      Branch_Type type;
      type.is_conditional = b.isConditional();
      type.is_indirect = b.isIndirect();
      impl.update_speculative_state(branchId, b.ip(), type, b.isTaken(),
                                    b.target());
      impl.commit_state_at_retire(branchId, b.ip(), type, b.isTaken(),
                                  b.target());
    }
    state = kNone;
  }

  mbp::json metadata_stats() const override {
    return {
        {"name", "Adapter of Scarab's TAGE-SC-L to MBPlib"},
    };
  }
};

}  // namespace tagescl

#endif  // TAGESCL_ZIP_HIST_MBPLIB_COMPAT_LAYER_HPP_
