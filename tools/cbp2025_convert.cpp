/* Convert a CBP2025 instruction-level trace into the project's two inputs:
 *
 *   1. an SBBT branch stream (.sbbt.zst), so the validated TAGE-SC-L driver
 *      (build/run_predictors), the dataset builder and the history window all
 *      run on CBP2025 data unchanged;
 *   2. optionally, per-instance FETCH-SAFE raw features for selected branches,
 *      joined later on branch_index exactly like the TAGE prediction log.
 *
 * Trace layout: the CBP2025 framework's lib/trace_reader.h (readInstr). One record
 * per trace instruction:
 *
 *   pc u64 | class u8 | [mem: ea u64, size u8, base_upd u8, [store: reg_off u8]]
 *   | [branch: taken u8, [taken: target u64]] | n_in u8, in_regs[n_in]
 *   | n_out u8, out_regs[n_out], out_values (8 B int, 16 B SIMD each)
 *
 * Registers: 0-30 GPR, 31 SP, 32-63 SIMD, 64 NZCV flags, 65 zero register.
 *
 * ---------------------------------------------------------------------------
 * THE FETCH-SAFETY RULE for features. A branch predictor answers at fetch. A
 * feature is admitted only if its value exists, in a real core, at the moment
 * this branch is fetched:
 *
 *   path          derived from the (predicted) branch path, the same
 *                 assumption every global-history predictor makes: call
 *                 depth (RAS), basic-block size, distance since this PC last
 *                 ran, this PC's local outcome history, indirect-target bits.
 *   inst_mix      instruction classes of OLDER instructions, ending
 *                 --decode-lag instructions before the branch, so they have
 *                 been decoded by the time it is fetched. ASSUMPTION: front
 *                 end depth x width <= decode-lag (default 64).
 *   committed     register values from the RETIREMENT register file only:
 *                 writes by instructions at least --commit-lag instructions
 *                 older than the branch. Fetch stalls once ROB + fetch queue
 *                 are full, so an instruction that far back must have retired.
 *                 ASSUMPTION: ROB + fetch queue <= commit-lag (default 1024;
 *                 the largest shipping ROBs are ~600-630 entries).
 *
 * EXCLUDED, deliberately: the branch's own source-register values (for B.cond
 * that is the NZCV result of the compare that decides it -- a perfect,
 * physically impossible predictor), its outcome and target, and any value or
 * address produced by an instruction that may still be in flight.
 * ---------------------------------------------------------------------------
 *
 * Build:
 *   clang++ -std=c++17 -O3 -I third_party/mbplib/include \
 *     tools/cbp2025_convert.cpp -lz -o build/cbp2025_convert
 */

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

}  // namespace

#include "cbp2025_trace.hpp"

namespace {
using namespace cbp25;


/* ------------------------------------------------------------------------ */
/* Feature state                                                             */
/* ------------------------------------------------------------------------ */

constexpr int kLocalBits = 16;
constexpr int kIndTargetBits = 8;
constexpr int kMixClasses = 5;   // load, store, fp, slow_alu, branch
constexpr int kMaxWrites = 4;

struct PcState {
  bool seen = false;
  uint64_t last_instr = 0;
  uint64_t last_branch = 0;
  uint32_t local = 0;       // bit 0 = most recent outcome
  uint32_t run = 0;         // length of the current same-outcome run
};

struct PendingWrites {
  uint8_t n = 0;
  uint8_t reg[kMaxWrites];
  uint64_t value[kMaxWrites];
};

std::vector<std::string> FeatureNames() {
  std::vector<std::string> n = {"call_depth", "bb_size", "instr_since_same_pc",
                                "branches_since_same_pc", "first_occurrence"};
  for (int i = 0; i < kLocalBits; ++i) n.push_back("local_hist_" + std::to_string(i));
  n.push_back("local_run_len");
  for (int i = 0; i < kIndTargetBits; ++i) n.push_back("ind_target_bit_" + std::to_string(i));
  for (const char* c : {"load", "store", "fp", "slow_alu", "branch"})
    n.push_back(std::string("mix_") + c);
  for (int s = 0; s < 2; ++s) {
    std::string p = "src" + std::to_string(s) + "_";
    for (const char* f : {"valid", "is_flags", "is_zero", "is_negative", "log_mag",
                          "bit0", "bit1", "bit2", "bit3", "commit_age"})
      n.push_back(p + f);
  }
  return n;
}

/* Feature group of each column, recorded in the sidecar so the Python side can
 * select groups for ablations. Must follow FeatureNames() order. */
std::vector<std::string> FeatureGroups() {
  std::vector<std::string> g;
  for (int i = 0; i < 5 + kLocalBits + 1 + kIndTargetBits; ++i) g.push_back("path");
  for (int i = 0; i < kMixClasses; ++i) g.push_back("inst_mix");
  for (int i = 0; i < 20; ++i) g.push_back("committed");
  return g;
}

float Log2Scaled(double v, double scale) {
  return static_cast<float>(std::log2(1.0 + v) / scale);
}

void Usage(const char* a0) {
  std::cerr << "usage: " << a0 << " <trace.gz> [options]\n"
            << "  --sbbt PATH           write the branch stream (.sbbt.zst via zstd)\n"
            << "  --meta PATH           write counts + split boundaries as JSON\n"
            << "  --splits a,b,...      fractions of the branch stream to record\n"
            << "                        boundaries at (default 0.6,0.7)\n"
            << "  --feature-pcs PATH    log fetch-safe features for these PCs\n"
            << "  --features PREFIX     write PREFIX.bin + PREFIX.json\n"
            << "  --commit-lag N        retirement-RF lag in instructions (1024)\n"
            << "  --decode-lag N        inst-mix lag in instructions (64)\n"
            << "  --mix-window N        inst-mix window in instructions (128)\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { Usage(argv[0]); return 2; }
  std::string trace = argv[1], sbbt_path, meta_path, feat_pcs_path, feat_prefix;
  std::vector<double> splits = {0.6, 0.7};
  uint64_t commit_lag = 1024, decode_lag = 64, mix_window = 128;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::cerr << a << " needs a value\n"; exit(2); }
      return argv[++i];
    };
    if (a == "--sbbt") sbbt_path = next();
    else if (a == "--meta") meta_path = next();
    else if (a == "--feature-pcs") feat_pcs_path = next();
    else if (a == "--features") feat_prefix = next();
    else if (a == "--commit-lag") commit_lag = std::stoull(next());
    else if (a == "--decode-lag") decode_lag = std::stoull(next());
    else if (a == "--mix-window") mix_window = std::stoull(next());
    else if (a == "--splits") {
      splits.clear();
      std::string s = next();
      size_t st = 0;
      while (st <= s.size()) {
        size_t c = s.find(',', st);
        splits.push_back(std::stod(s.substr(st, c == std::string::npos ? c : c - st)));
        if (c == std::string::npos) break;
        st = c + 1;
      }
    } else { Usage(argv[0]); return 2; }
  }
  if (feat_pcs_path.empty() != feat_prefix.empty()) {
    std::cerr << "--feature-pcs and --features go together\n";
    return 2;
  }

  /* Pass 1: totals, and the taken target of every conditional branch. CBP2025
   * records a target only when the branch is taken, but SBBT (and TAGE-SC-L's
   * path history) expects a conditional branch's target either way. A direct
   * conditional branch has one target, so the first taken instance supplies
   * it; a branch never taken in the trace falls back to pc+4. */
  std::unordered_map<uint64_t, uint64_t> cond_target;
  uint64_t n_instr = 0, n_branch = 0, n_cond = 0;
  {
    GzReader in(trace);
    Instr r;
    while (ReadInstr(in, r)) {
      ++n_instr;
      if (!IsBranch(r.cls)) continue;
      ++n_branch;
      if (r.cls == kCond) {
        ++n_cond;
        if (r.taken) cond_target.emplace(r.pc, r.target);
      }
    }
  }
  std::cerr << trace << ": " << n_instr << " instructions, " << n_branch
            << " branches, " << n_cond << " conditional\n";

  // Split boundaries, in absolute branch-index units.
  std::vector<uint64_t> boundary;
  for (double f : splits) boundary.push_back(static_cast<uint64_t>(f * n_branch));
  std::vector<uint64_t> boundary_instr(splits.size(), 0), boundary_cond(splits.size(), 0);

  const bool want_features = !feat_prefix.empty();
  std::unordered_set<uint64_t> feat_pcs;
  if (want_features) feat_pcs = ReadPcList(feat_pcs_path);

  FILE* sbbt = nullptr;
  if (!sbbt_path.empty()) {
    std::string cmd = "zstd -q -f -T0 -o '" + sbbt_path + "'";
    sbbt = popen(cmd.c_str(), "w");
    if (!sbbt) { std::cerr << "cannot start zstd\n"; return 1; }
    const uint64_t header[3] = {0x0000010A54424253ull, n_instr, n_branch};
    fwrite(header, 8, 3, sbbt);
  }

  const auto names = FeatureNames();
  const size_t n_feat = names.size();
  std::ofstream feat_out;
  if (want_features) {
    feat_out.open(feat_prefix + ".bin", std::ios::binary);
    if (!feat_out) { std::cerr << "cannot write features\n"; return 1; }
  }
  std::vector<char> rec(16 + 4 * n_feat);
  uint64_t feat_records = 0;

  // Feature state.
  std::unordered_map<uint64_t, PcState> pc_state;
  int64_t call_depth = 0;
  uint64_t last_ind_target = 0;
  uint64_t last_branch_instr = 0;          // instruction number of previous branch
  std::vector<uint8_t> cls_ring(decode_lag + mix_window, 255);
  std::array<int64_t, kMixClasses> mix_count{};
  std::vector<PendingWrites> rf_ring(commit_lag);
  std::array<uint64_t, 66> committed_value{};
  std::array<int64_t, 66> committed_at;
  committed_at.fill(-1);

  auto mix_slot = [](uint8_t c) -> int {
    switch (c) {
      case kLoad: return 0;
      case kStore: return 1;
      case kFp: return 2;
      case kSlowAlu: return 3;
      default: return IsBranch(c) ? 4 : -1;
    }
  };

  /* Pass 2. Instruction numbers are 1-based, matching SBBT's cumulative
   * ninstr (MBPlib's instr_num for a branch includes the branch itself). */
  GzReader in(trace);
  Instr r;
  uint64_t instr_no = 0, branch_idx = 0, cond_seen = 0, sbbt_instr = 0;
  size_t next_boundary = 0;
  while (ReadInstr(in, r)) {
    ++instr_no;
    const uint64_t pos = instr_no - 1;  // 0-based ring position

    if (want_features) {
      /* Retire the instruction falling out of the commit-lag window into the
       * retirement register file BEFORE looking at this instruction, so the
       * committed RF holds exactly the writes of instructions at least
       * commit_lag older than the current one. */
      PendingWrites& old = rf_ring[pos % commit_lag];
      if (pos >= commit_lag) {
        for (int k = 0; k < old.n; ++k) {
          committed_value[old.reg[k]] = old.value[k];
          committed_at[old.reg[k]] = static_cast<int64_t>(pos - commit_lag);
        }
      }
      old.n = 0;

      // Slide the decode-lagged instruction-mix window.
      const size_t ring = cls_ring.size();
      if (pos >= decode_lag) {
        int s = mix_slot(cls_ring[(pos - decode_lag) % ring]);
        if (s >= 0) ++mix_count[s];   // enters the window
      }
      if (pos >= decode_lag + mix_window) {
        int s = mix_slot(cls_ring[(pos - decode_lag - mix_window) % ring]);
        if (s >= 0) --mix_count[s];   // leaves the window
      }
    }

    if (IsBranch(r.cls)) {
      while (next_boundary < boundary.size() &&
             branch_idx == boundary[next_boundary]) {
        boundary_instr[next_boundary] = sbbt_instr + std::min<uint64_t>(
            instr_no - last_branch_instr, 4095);   // instr_num of THIS branch
        boundary_cond[next_boundary] = cond_seen;
        ++next_boundary;
      }

      const bool cond = r.cls == kCond;
      uint64_t target = r.target;
      if (cond && !r.taken) {
        auto it = cond_target.find(r.pc);
        target = it != cond_target.end() ? it->second : r.pc + 4;
      }
      const uint64_t gap = instr_no - last_branch_instr;

      if (want_features && cond && feat_pcs.count(r.pc)) {
        PcState& st = pc_state[r.pc];
        float* f = reinterpret_cast<float*>(rec.data() + 16);
        size_t k = 0;
        f[k++] = static_cast<float>(std::clamp<int64_t>(call_depth, 0, 255)) / 64.0f;
        f[k++] = Log2Scaled(static_cast<double>(gap), 12.0);
        f[k++] = st.seen ? Log2Scaled(double(instr_no - st.last_instr), 32.0) : 1.0f;
        f[k++] = st.seen ? Log2Scaled(double(branch_idx - st.last_branch), 32.0) : 1.0f;
        f[k++] = st.seen ? 0.0f : 1.0f;
        for (int b = 0; b < kLocalBits; ++b) f[k++] = float((st.local >> b) & 1u);
        f[k++] = Log2Scaled(st.run, 16.0);
        uint64_t h = last_ind_target >> 2;
        h ^= h >> 8; h ^= h >> 16; h ^= h >> 32;
        for (int b = 0; b < kIndTargetBits; ++b) f[k++] = float((h >> b) & 1u);
        const double win = static_cast<double>(std::max<uint64_t>(
            1, std::min<uint64_t>(mix_window, pos > decode_lag ? pos - decode_lag : 0)));
        for (int c = 0; c < kMixClasses; ++c) f[k++] = float(mix_count[c] / win);
        for (int s = 0; s < 2; ++s) {
          uint8_t reg = 0;
          bool valid = false;
          // Source slot s among the branch's integer input registers.
          int seen_int = 0;
          for (int q = 0; q < r.n_in; ++q) {
            if (!IsIntReg(r.in_regs[q])) continue;
            if (seen_int++ == s) { reg = r.in_regs[q]; valid = true; break; }
          }
          const bool is_flags = valid && reg == 64;
          uint64_t v = 0;
          int64_t at = -1;
          if (valid && reg != 65) { v = committed_value[reg]; at = committed_at[reg]; }
          if (valid && reg == 65) { at = 0; }      // zero register: always 0
          const bool known = valid && at >= 0;
          f[k++] = known ? 1.0f : 0.0f;
          f[k++] = is_flags ? 1.0f : 0.0f;
          f[k++] = known && v == 0 ? 1.0f : 0.0f;
          f[k++] = known && !is_flags && (v >> 63) ? 1.0f : 0.0f;
          const double mag = static_cast<double>(
              (v >> 63) && !is_flags ? ~v + 1 : v);
          f[k++] = known && !is_flags ? Log2Scaled(mag, 64.0) : 0.0f;
          // NZCV live in bits 31..28; elsewhere use the low four bits.
          const uint64_t nib = is_flags ? (v >> 28) & 0xF : v & 0xF;
          for (int b = 0; b < 4; ++b) f[k++] = known ? float((nib >> b) & 1u) : 0.0f;
          f[k++] = known && reg != 65
                       ? Log2Scaled(double(pos - static_cast<uint64_t>(at)), 32.0)
                       : 0.0f;
        }
        if (k != n_feat) { std::cerr << "feature count mismatch\n"; return 1; }
        const int64_t bi = static_cast<int64_t>(branch_idx);
        std::memcpy(rec.data(), &bi, 8);
        std::memcpy(rec.data() + 8, &r.pc, 8);
        feat_out.write(rec.data(), static_cast<std::streamsize>(rec.size()));
        ++feat_records;

        // Update local state AFTER the features are taken.
        const bool last = st.local & 1u;
        st.run = (st.seen && last == r.taken) ? st.run + 1 : 1;
        st.local = (st.local << 1) | (r.taken ? 1u : 0u);
        st.seen = true;
        st.last_instr = instr_no;
        st.last_branch = branch_idx;
      }

      if (sbbt) {
        const uint64_t ninstr = std::min<uint64_t>(gap, 4095);
        sbbt_instr += ninstr;
        const uint64_t w0 = uint64_t(SbbtOpcode(r.cls)) |
                            (uint64_t(r.taken || !cond) << 11) |
                            ((r.pc & kIpMask) << 12);
        const uint64_t w1 = ninstr | ((target & kIpMask) << 12);
        const uint64_t w[2] = {w0, w1};
        fwrite(w, 8, 2, sbbt);
      } else {
        sbbt_instr += std::min<uint64_t>(gap, 4095);
      }

      // Path state, updated after this branch's own features.
      if (r.cls == kCallDirect || r.cls == kCallIndirect) ++call_depth;
      if (r.cls == kReturn) --call_depth;
      if (r.cls == kUncondIndirect || r.cls == kCallIndirect || r.cls == kReturn)
        last_ind_target = target;
      if (cond) ++cond_seen;
      last_branch_instr = instr_no;
      ++branch_idx;
    }

    if (want_features) {
      cls_ring[pos % cls_ring.size()] = r.cls;
      PendingWrites& w = rf_ring[pos % commit_lag];
      for (int q = 0; q < r.n_out && w.n < kMaxWrites; ++q) {
        const uint8_t reg = r.out_regs[q];
        if (IsIntReg(reg) && reg != 65) {
          w.reg[w.n] = reg; w.value[w.n] = r.out_int_values[q]; ++w.n;
        }
      }
    }
  }
  // Boundaries at or past the end (fraction 1.0).
  while (next_boundary < boundary.size()) {
    boundary_instr[next_boundary] = sbbt_instr;
    boundary_cond[next_boundary] = cond_seen;
    ++next_boundary;
  }

  if (sbbt && pclose(sbbt) != 0) { std::cerr << "zstd failed\n"; return 1; }
  if (instr_no != n_instr || branch_idx != n_branch) {
    std::cerr << "pass 2 disagrees with pass 1\n";
    return 1;
  }

  if (want_features) {
    feat_out.close();
    json formats = json::array({"<i8", "<u8"});
    json fnames = json::array({"branch_index", "pc"});
    json sidecar = {
        {"format", "cbp2025-fetch-safe-features"}, {"version", 1},
        {"trace", trace}, {"records", feat_records},
        {"record_bytes", rec.size()},
        {"feature_names", names}, {"feature_groups", FeatureGroups()},
        {"feature_dtype", "<f4"},
        {"commit_lag", commit_lag}, {"decode_lag", decode_lag},
        {"mix_window", mix_window},
        {"note", "record = branch_index <i8, pc <u8, then len(feature_names) "
                 "float32. branch_index is the position among ALL branches, "
                 "matching the SBBT stream and the TAGE prediction log."},
    };
    std::ofstream fh(feat_prefix + ".json");
    fh << sidecar.dump(2) << '\n';
    std::cerr << "wrote " << feat_records << " feature records\n";
  }

  if (!meta_path.empty()) {
    json b = json::array();
    for (size_t i = 0; i < splits.size(); ++i)
      b.push_back({{"fraction", splits[i]}, {"branch_index", boundary[i]},
                   {"instr_num", boundary_instr[i]},
                   {"conditional_before", boundary_cond[i]}});
    json meta = {{"trace", trace}, {"num_instructions", n_instr},
                 {"num_branches", n_branch}, {"num_conditional", n_cond},
                 {"sbbt_instr_sum", sbbt_instr},
                 {"distinct_conditional_pcs_taken", cond_target.size()},
                 {"boundaries", b}};
    std::ofstream fh(meta_path);
    fh << meta.dump(2) << '\n';
  }
  return 0;
}
