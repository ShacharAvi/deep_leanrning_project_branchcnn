/* Run every reference predictor over an SBBT trace in a single pass.
 *
 * Spec baselines 0-5. Where MBPlib ships a validated implementation we use it
 * (with MBPlib's own documented 64 KB parameters, from example/CMakeLists.txt)
 * rather than writing our own; the rest live in predictors_extra.hpp.
 *
 * Simulation protocol, matching mbp::Simulate:
 *
 *     if (branch.isConditional()) { p->predict(ip); p->train(branch); }
 *     p->track(branch);
 *
 * track() after train() is a no-op for the TAGE-SC-L adapter (its state is
 * already kCommited), so one uniform loop is correct for every predictor.
 *
 * Build (no cmake needed):
 *   clang++ -std=c++17 -O3 -DTAGE_SC_L_SIZE=64 \
 *     -I third_party/mbplib/include -I third_party/spec_tage_scl/include \
 *     tools/run_predictors.cpp third_party/mbplib/src/{core,sim}/ *.cpp \
 *     -o build/run_predictors
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mbp/examples/bimodal.hpp>
#include <mbp/examples/gshare.hpp>
#include <mbp/examples/hashed_perceptron.hpp>
#include <mbp/sim/sbbt_reader.hpp>
#include <nlohmann/json.hpp>

#include "predictors_extra.hpp"
#include "tagescl/ifaces/mbplib/tage_sc_l_sim_only.hpp"

using json = mbp::json;

/* MBPlib derives HashedPerceptron's history length from a geometric series; we
 * patched that computation to be constexpr under Apple clang (see
 * mbp/examples/hashed_perceptron.hpp). Pin the result so any drift in that
 * patch fails the build rather than silently changing the predictor. */
static_assert(mbp::HashedPerceptron<4, 16, 12, 18>::H == 259,
              "HashedPerceptron history length changed -- check the constexpr "
              "sqrt/pow patch in mbp/examples/hashed_perceptron.hpp");

namespace {

struct Entry {
  std::string name;
  std::unique_ptr<mbp::Predictor> predictor;
  int64_t mispredictions = 0;
};

/* Per-PC aggregate of TAGE-SC-L's own internal state at prediction time.
 *
 * Everything here is computed inside get_prediction(), so it is available at
 * fetch in a real pipeline -- recording it is not a leak. The point is to
 * classify *why* a branch is hard (project spec section 2.3) before training
 * anything:
 *
 *   hit_bank      which history length supplied the prediction. Bank 0 means no
 *                 tagged component matched and TAGE fell back to bimodal, i.e.
 *                 no history pattern of any length helped. High banks mean the
 *                 correlation is real but very deep.
 *   confidence    TAGE's own 3-level difficulty estimate.
 *   alt_disagree  provider and alternate disagree -> the predictor is unsure.
 *   sc_margin     |statistical corrector sum| relative to its threshold.
 *
 * Aggregated per PC rather than logged per instance: classification only needs
 * the distribution, and per-instance records would cost gigabytes. If the
 * ablation phase later feeds this to a model, log it per instance then.
 */
struct TageStateAccum {
  int64_t samples = 0;
  int64_t hit_bank_sum = 0;
  int64_t bimodal_fallback = 0;   // hit_bank == 0
  int64_t high_conf = 0;
  int64_t med_conf = 0;
  int64_t low_conf = 0;
  int64_t alt_disagree = 0;
  int64_t loop_valid = 0;
  double sc_margin_sum = 0.0;
  int max_hit_bank = 0;
};

struct PcStats {
  int64_t occurrences = 0;
  int64_t taken = 0;
  std::vector<int64_t> misses;  // one per predictor
  TageStateAccum tage;
};

/* One row of the prediction log: what happened at a single dynamic instance of
 * a selected branch. Packed, so numpy can map it directly with
 *   np.dtype([('branch_index','<i8'),('pc','<u8'),('taken','u1'),
 *             ('mispredicted','u1'),('hit_bank','i1'),('alt_bank','i1'),
 *             ('tage_flags','u1'),('sc_sum','<i2'),('sc_threshold','<i2')],
 *            align=False)
 * which has itemsize 25. Keep the two in sync -- there is a test for it. */
#pragma pack(push, 1)
struct LogRecord {
  int64_t branch_index;   // absolute index among ALL branches in the trace
  uint64_t pc;
  uint8_t taken;
  uint8_t mispredicted;   // bit i set == predictor i was wrong

  /* TAGE-SC-L state at THIS prediction. All of it is computed inside
   * get_prediction(), so it is available at fetch in a real pipeline -- these
   * are legitimate model inputs, unlike the per-PC aggregates in the CSV which
   * are constants inside a per-branch model.
   *
   * Zero-filled when TAGE-SC-L is not among the selected predictors; the
   * sidecar's has_tage_state flag says which. */
  int8_t hit_bank;        // which tagged component answered; 0 == bimodal fallback
  int8_t alt_bank;
  uint8_t tage_flags;     // see TageFlag below
  int16_t sc_sum;         // statistical corrector weighted sum
  int16_t sc_threshold;   // its adaptive threshold
};
#pragma pack(pop)
static_assert(sizeof(LogRecord) == 25, "LogRecord must stay packed at 25 bytes");

enum TageFlag : uint8_t {
  kHighConfidence = 1u << 0,
  kMediumConfidence = 1u << 1,
  kLowConfidence = 1u << 2,
  kLongestMatchPrediction = 1u << 3,
  kAltPrediction = 1u << 4,
  kLoopValid = 1u << 5,
  kFinalPrediction = 1u << 6,   // what TAGE-SC-L actually output
  kTagePrediction = 1u << 7,    // the TAGE component alone, pre-corrector
};

using TageScl = tagescl::MbpTageScl<tagescl::CONFIG_64KB>;

/* MBPlib's documented 64 KB configurations (example/CMakeLists.txt), so every
 * learning predictor here gets the same storage budget -- the budget-matched
 * comparison the spec asks for in section 15.
 *
 * ``tage_out`` receives a typed handle on the TAGE-SC-L instance (if selected)
 * so its internal prediction state can be read; the generic mbp::Predictor
 * interface does not expose it. */
std::vector<Entry> BuildPredictors(const std::vector<std::string>& wanted,
                                   TageScl** tage_out) {
  std::vector<Entry> all;
  auto add = [&](const char* name, mbp::Predictor* p) {
    all.push_back({name, std::unique_ptr<mbp::Predictor>(p), 0});
  };

  auto* tage = new TageScl(1);
  add("always_taken", new bp::AlwaysTaken{});
  add("always_not_taken", new bp::AlwaysNotTaken{});
  add("bimodal_64KB", new mbp::Bimodal<18>{});
  add("gshare_64KB", new mbp::Gshare<25, 18>{});
  add("perceptron_64KB", new bp::ClassicPerceptron<62, 10, 8>{});
  add("hashed_perceptron_64KB", new mbp::HashedPerceptron<4, 16, 12, 18>{});
  add("tage_sc_l_64KB", tage);

  *tage_out = nullptr;
  if (wanted.empty()) {
    *tage_out = tage;
    return all;
  }

  std::vector<Entry> selected;
  for (const auto& name : wanted) {
    auto it = std::find_if(all.begin(), all.end(),
                           [&](const Entry& e) { return e.name == name; });
    if (it == all.end()) {
      std::cerr << "unknown predictor: " << name << "\navailable:";
      for (const auto& e : all) std::cerr << ' ' << e.name;
      std::cerr << '\n';
      exit(2);
    }
    if (it->name == "tage_sc_l_64KB") *tage_out = tage;
    selected.push_back(std::move(*it));
  }
  return selected;
}

std::vector<std::string> SplitCommas(const std::string& s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t comma = s.find(',', start);
    if (comma == std::string::npos) {
      out.push_back(s.substr(start));
      break;
    }
    out.push_back(s.substr(start, comma - start));
    start = comma + 1;
  }
  return out;
}

/* Read a hard-branch file: one hexadecimal PC per line, '#' comments allowed.
 * Same shape as BranchNet's hard_brs files, so those can be used directly. */
std::unordered_set<uint64_t> ReadPcList(const std::string& path) {
  std::ifstream fh(path);
  if (!fh) {
    std::cerr << "cannot read PC list " << path << '\n';
    exit(2);
  }
  std::unordered_set<uint64_t> pcs;
  std::string line;
  while (std::getline(fh, line)) {
    size_t hash = line.find('#');
    if (hash != std::string::npos) line.resize(hash);
    size_t begin = line.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) continue;
    size_t end = line.find_last_not_of(" \t\r\n");
    pcs.insert(std::stoull(line.substr(begin, end - begin + 1), nullptr, 16));
  }
  if (pcs.empty()) {
    std::cerr << "PC list " << path << " is empty\n";
    exit(2);
  }
  return pcs;
}

void Usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0 << " <trace.sbbt[.zst]> [options]\n"
      << "  --warmup N        instructions to simulate before counting\n"
      << "  --limit N         stop after N conditional branches\n"
      << "  --predictors a,b  subset to run (default: all)\n"
      << "  --json PATH       write the summary as JSON (default: stdout)\n"
      << "  --per-pc PATH     write per-PC per-predictor stats as CSV\n"
      << "  --min-occurrences N  omit PCs below this count from --per-pc "
         "(default 1)\n"
      << "\n"
      << "  Prediction log -- per-instance outcomes, for hybrid evaluation:\n"
      << "  --log PREFIX      write PREFIX.bin + PREFIX.json\n"
      << "  --log-pcs PATH    restrict the log to these PCs (hex, one per "
         "line).\n"
      << "                    Strongly recommended: logging every conditional\n"
      << "                    branch costs ~18 bytes each (>5 GB on a large "
         "trace).\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    Usage(argv[0]);
    return 2;
  }

  std::string tracepath = argv[1];
  int64_t warmup_instrs = 0;
  int64_t limit_branches = -1;
  int64_t min_occurrences = 1;
  std::string json_path, per_pc_path, log_prefix, log_pcs_path;
  std::vector<std::string> wanted;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << arg << " requires a value\n";
        exit(2);
      }
      return argv[++i];
    };
    if (arg == "--warmup") warmup_instrs = std::stoll(next());
    else if (arg == "--limit") limit_branches = std::stoll(next());
    else if (arg == "--min-occurrences") min_occurrences = std::stoll(next());
    else if (arg == "--predictors") wanted = SplitCommas(next());
    else if (arg == "--json") json_path = next();
    else if (arg == "--per-pc") per_pc_path = next();
    else if (arg == "--log") log_prefix = next();
    else if (arg == "--log-pcs") log_pcs_path = next();
    else if (arg == "-h" || arg == "--help") { Usage(argv[0]); return 0; }
    else { std::cerr << "unknown option: " << arg << '\n'; Usage(argv[0]); return 2; }
  }

  if (!log_pcs_path.empty() && log_prefix.empty()) {
    std::cerr << "--log-pcs requires --log\n";
    return 2;
  }

  /* --log PREFIX writes PREFIX.json; refuse to let it clobber --json silently. */
  if (!log_prefix.empty() && !json_path.empty() &&
      json_path == log_prefix + ".json") {
    std::cerr << "--json and --log would both write " << json_path
              << "; give them different paths\n";
    return 2;
  }
  if (!log_prefix.empty() && !per_pc_path.empty() &&
      per_pc_path == log_prefix + ".bin") {
    std::cerr << "--per-pc and --log would both write " << per_pc_path
              << "; give them different paths\n";
    return 2;
  }

  TageScl* tage = nullptr;
  auto predictors = BuildPredictors(wanted, &tage);
  const size_t num_predictors = predictors.size();

  // Index of TAGE-SC-L among the running predictors, so the state we record is
  // attributed to the right one.
  size_t tage_index = num_predictors;
  for (size_t i = 0; i < num_predictors; ++i) {
    if (predictors[i].name == "tage_sc_l_64KB") tage_index = i;
  }
  if (num_predictors > 8 && !log_prefix.empty()) {
    std::cerr << "the prediction log packs mispredictions into one byte, so it "
                 "supports at most 8 predictors (got "
              << num_predictors << ")\n";
    return 2;
  }

  // Prediction log setup.
  std::unordered_set<uint64_t> log_pcs;
  const bool log_filtered = !log_pcs_path.empty();
  if (log_filtered) log_pcs = ReadPcList(log_pcs_path);

  std::ofstream log_file;
  std::vector<LogRecord> log_buffer;
  int64_t log_records = 0;
  if (!log_prefix.empty()) {
    log_file.open(log_prefix + ".bin", std::ios::binary);
    if (!log_file) {
      std::cerr << "cannot write " << log_prefix << ".bin\n";
      return 1;
    }
    log_buffer.reserve(1 << 16);
    if (!log_filtered) {
      std::cerr << "warning: --log without --log-pcs records every conditional "
                   "branch (~18 bytes each)\n";
    }
  }
  auto flush_log = [&]() {
    if (!log_buffer.empty()) {
      log_file.write(reinterpret_cast<const char*>(log_buffer.data()),
                     static_cast<std::streamsize>(log_buffer.size() *
                                                  sizeof(LogRecord)));
      log_buffer.clear();
    }
  };

  mbp::SbbtReader trace{tracepath};
  std::unordered_map<uint64_t, PcStats> per_pc;
  int64_t total_branches = 0;      // all branches seen
  int64_t counted_branches = 0;    // conditional, post-warmup
  int64_t skipped_warmup = 0;

  const auto start = std::chrono::high_resolution_clock::now();

  // Scratch for the current branch's TAGE state, filled between predict() and
  // train() where lastPredictionInfo() is valid.
  struct LoggedTage {
    int hit_bank, alt_bank, sc_sum, sc_threshold;
    uint8_t flags;
  } logged_tage{};
  bool logged_tage_valid = false;

  mbp::Branch b;
  int64_t instr_num = 0;
  while ((instr_num = trace.nextBranch(b)) !=
         std::numeric_limits<int64_t>::max()) {
    const int64_t branch_index = total_branches;
    ++total_branches;

    if (b.isConditional()) {
      logged_tage_valid = false;
      const bool counted = instr_num >= warmup_instrs;
      PcStats* stats = nullptr;

      if (counted) {
        ++counted_branches;
        stats = &per_pc[b.ip()];
        if (stats->misses.empty()) stats->misses.assign(num_predictors, 0);
        ++stats->occurrences;
        stats->taken += b.isTaken();
      } else {
        ++skipped_warmup;
      }

      uint8_t mispredicted_mask = 0;
      for (size_t i = 0; i < num_predictors; ++i) {
        const bool predicted = predictors[i].predictor->predict(b.ip());

        /* Read TAGE's internal state between predict() and train() -- it is
         * only valid in that window, and it reflects what the predictor knew
         * at fetch time. */
        if (i == tage_index && tage != nullptr) {
          const auto& pi = tage->lastPredictionInfo();
          logged_tage_valid = true;
          logged_tage.hit_bank = pi.tage.hit_bank;
          logged_tage.alt_bank = pi.tage.alt_bank;
          logged_tage.sc_sum = pi.sc.gehls_sum;
          logged_tage.sc_threshold = pi.sc.thresholds_sum;
          logged_tage.flags = static_cast<uint8_t>(
              (pi.tage.high_confidence ? kHighConfidence : 0) |
              (pi.tage.medium_confidence ? kMediumConfidence : 0) |
              (pi.tage.low_confidence ? kLowConfidence : 0) |
              (pi.tage.longest_match_prediction ? kLongestMatchPrediction : 0) |
              (pi.tage.alt_prediction ? kAltPrediction : 0) |
              (pi.loop.valid ? kLoopValid : 0) |
              (pi.final_prediction ? kFinalPrediction : 0) |
              (pi.tage.prediction ? kTagePrediction : 0));
        }
        if (i == tage_index && counted && tage != nullptr) {
          const auto& pi = tage->lastPredictionInfo();
          auto& acc = stats->tage;
          ++acc.samples;
          acc.hit_bank_sum += pi.tage.hit_bank;
          acc.max_hit_bank = std::max(acc.max_hit_bank, pi.tage.hit_bank);
          acc.bimodal_fallback += (pi.tage.hit_bank == 0);
          acc.high_conf += pi.tage.high_confidence;
          acc.med_conf += pi.tage.medium_confidence;
          acc.low_conf += pi.tage.low_confidence;
          acc.alt_disagree +=
              (pi.tage.longest_match_prediction != pi.tage.alt_prediction);
          acc.loop_valid += pi.loop.valid;
          if (pi.sc.thresholds_sum != 0) {
            acc.sc_margin_sum +=
                std::abs(static_cast<double>(pi.sc.gehls_sum)) /
                pi.sc.thresholds_sum;
          }
        }

        predictors[i].predictor->train(b);
        if (predicted != b.isTaken()) {
          mispredicted_mask |= static_cast<uint8_t>(1u << i);
          if (counted) {
            ++predictors[i].mispredictions;
            ++stats->misses[i];
          }
        }
      }

      if (log_file.is_open() && counted &&
          (!log_filtered || log_pcs.count(b.ip()))) {
        LogRecord record{branch_index, b.ip(),
                         static_cast<uint8_t>(b.isTaken()), mispredicted_mask,
                         0, 0, 0, 0, 0};
        if (logged_tage_valid) {
          record.hit_bank = static_cast<int8_t>(logged_tage.hit_bank);
          record.alt_bank = static_cast<int8_t>(logged_tage.alt_bank);
          record.sc_sum = static_cast<int16_t>(std::clamp(
              logged_tage.sc_sum, -32768, 32767));
          record.sc_threshold = static_cast<int16_t>(std::clamp(
              logged_tage.sc_threshold, -32768, 32767));
          record.tage_flags = logged_tage.flags;
        }
        log_buffer.push_back(record);
        ++log_records;
        if (log_buffer.size() == log_buffer.capacity()) flush_log();
      }

      if (limit_branches > 0 && counted_branches >= limit_branches) {
        for (auto& e : predictors) e.predictor->track(b);
        break;
      }
    }

    for (auto& e : predictors) e.predictor->track(b);
  }

  if (log_file.is_open()) {
    flush_log();
    log_file.close();
  }

  const double elapsed =
      std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - start)
          .count();

  /* MPKI denominator. The header instruction count is authoritative: summed
   * per-branch ninstr omits the instructions trailing the final branch, and
   * ninstr is a saturating 12-bit field. When --limit cuts the run short we
   * fall back to the instruction number actually reached. */
  const bool exhausted = limit_branches <= 0;
  const int64_t metric_instrs =
      (exhausted ? static_cast<int64_t>(trace.numInstructions())
                 : trace.lastInstrRead()) -
      warmup_instrs;

  json predictors_json = json::array();
  for (const auto& e : predictors) {
    predictors_json.push_back({
        {"name", e.name},
        {"mispredictions", e.mispredictions},
        {"misprediction_rate",
         static_cast<double>(e.mispredictions) / counted_branches},
        {"accuracy",
         static_cast<double>(counted_branches - e.mispredictions) /
             counted_branches},
        {"mpki", 1000.0 * e.mispredictions / metric_instrs},
        {"config", e.predictor->metadata_stats()},
    });
  }

  json out = {
      {"metadata",
       {
           {"trace", tracepath},
           {"trace_instructions", trace.numInstructions()},
           {"trace_branches", trace.numBranches()},
           {"branches_read", total_branches},
           {"conditional_branches_counted", counted_branches},
           {"conditional_branches_skipped_as_warmup", skipped_warmup},
           {"warmup_instructions", warmup_instrs},
           {"metric_instructions", metric_instrs},
           {"exhausted_trace", exhausted},
           {"distinct_conditional_pcs", static_cast<int64_t>(per_pc.size())},
           {"simulation_seconds", elapsed},
       }},
      {"predictors", predictors_json},
  };

  if (json_path.empty()) {
    std::cout << out.dump(2) << '\n';
  } else {
    std::ofstream fh(json_path);
    if (!fh) { std::cerr << "cannot write " << json_path << '\n'; return 1; }
    fh << out.dump(2) << '\n';
    std::cerr << "wrote " << json_path << '\n';
  }

  if (!per_pc_path.empty()) {
    std::ofstream fh(per_pc_path);
    if (!fh) { std::cerr << "cannot write " << per_pc_path << '\n'; return 1; }

    const bool with_tage_state = (tage_index < num_predictors);

    fh << "pc,executions,taken,taken_rate";
    for (const auto& e : predictors) {
      fh << ',' << e.name << "_mispredictions"
         << ',' << e.name << "_mispredict_rate"
         << ',' << e.name << "_mpki";
    }
    if (with_tage_state) {
      fh << ",tage_hit_bank_mean,tage_hit_bank_max,tage_bimodal_frac"
         << ",tage_conf_high_frac,tage_conf_med_frac,tage_conf_low_frac"
         << ",tage_alt_disagree_frac,tage_loop_frac,tage_sc_margin_mean";
    }
    fh << '\n';

    std::vector<std::pair<uint64_t, const PcStats*>> rows;
    rows.reserve(per_pc.size());
    for (const auto& [pc, stats] : per_pc) {
      if (stats.occurrences >= min_occurrences) rows.emplace_back(pc, &stats);
    }
    // Most-mispredicted first, by the strongest predictor -- the H2P ranking.
    const size_t rank_by = num_predictors - 1;
    std::sort(rows.begin(), rows.end(), [&](const auto& l, const auto& r) {
      if (l.second->misses[rank_by] != r.second->misses[rank_by])
        return l.second->misses[rank_by] > r.second->misses[rank_by];
      return l.first < r.first;
    });

    for (const auto& [pc, stats] : rows) {
      fh << "0x" << std::hex << pc << std::dec
         << ',' << stats->occurrences
         << ',' << stats->taken
         << ',' << static_cast<double>(stats->taken) / stats->occurrences;
      for (size_t i = 0; i < num_predictors; ++i) {
        const int64_t misses = stats->misses[i];
        fh << ',' << misses
           << ',' << static_cast<double>(misses) / stats->occurrences
           << ',' << 1000.0 * misses / metric_instrs;
      }
      if (with_tage_state) {
        const auto& a = stats->tage;
        const double n = a.samples ? static_cast<double>(a.samples) : 1.0;
        fh << ',' << a.hit_bank_sum / n
           << ',' << a.max_hit_bank
           << ',' << a.bimodal_fallback / n
           << ',' << a.high_conf / n
           << ',' << a.med_conf / n
           << ',' << a.low_conf / n
           << ',' << a.alt_disagree / n
           << ',' << a.loop_valid / n
           << ',' << a.sc_margin_sum / n;
      }
      fh << '\n';
    }
    std::cerr << "wrote " << per_pc_path << " (" << rows.size() << " PCs)\n";
  }

  if (!log_prefix.empty()) {
    json names = json::array();
    for (const auto& e : predictors) names.push_back(e.name);

    /* Sidecar describing the .bin so a reader never has to guess the layout.
     * The record is packed; see LogRecord above. */
    json sidecar = {
        {"format", "branch-prediction-log"},
        {"version", 2},
        {"trace", tracepath},
        {"records", log_records},
        {"record_bytes", static_cast<int>(sizeof(LogRecord))},
        {"numpy_dtype",
         {{"names", {"branch_index", "pc", "taken", "mispredicted",
                     "hit_bank", "alt_bank", "tage_flags",
                     "sc_sum", "sc_threshold"}},
          {"formats", {"<i8", "<u8", "u1", "u1",
                       "i1", "i1", "u1", "<i2", "<i2"}},
          {"align", false}}},
        {"has_tage_state", tage_index < num_predictors},
        {"tage_flag_bits",
         {{"high_confidence", 1}, {"medium_confidence", 2},
          {"low_confidence", 4}, {"longest_match_prediction", 8},
          {"alt_prediction", 16}, {"loop_valid", 32},
          {"loop_prediction", 64}, {"tage_prediction", 128}}},
        {"predictors", names},
        {"mispredicted_bit_order", names},
        {"filtered_to_pcs", log_filtered},
        {"num_filter_pcs", log_filtered ? static_cast<int64_t>(log_pcs.size()) : -1},
        {"warmup_instructions", warmup_instrs},
        {"metric_instructions", metric_instrs},
        {"conditional_branches_counted", counted_branches},
        {"note",
         "branch_index is the absolute position among ALL branches in the "
         "trace (conditional and not), matching src.traces chunk.index. "
         "mispredicted is a bitmask: bit i corresponds to predictors[i]."},
    };
    std::ofstream fh(log_prefix + ".json");
    if (!fh) {
      std::cerr << "cannot write " << log_prefix << ".json\n";
      return 1;
    }
    fh << sidecar.dump(2) << '\n';
    std::cerr << "wrote " << log_prefix << ".bin + .json (" << log_records
              << " records, "
              << (log_records * sizeof(LogRecord)) / (1024.0 * 1024.0)
              << " MiB)\n";
  }

  return 0;
}
