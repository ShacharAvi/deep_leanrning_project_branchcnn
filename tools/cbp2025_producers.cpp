/* Dataflow "producers" of hard branches, and fetch-safe features from them.
 *
 *   profile   for each target branch, walk its dataflow backwards at every
 *             execution (branch <- compare <- load <- address ...), and count
 *             which static instructions appear in the chain
 *   features  per execution of each target branch, emit:
 *               producers  committed history of its chosen producer PCs
 *               inst_hist  types of the last 8/16/64 instructions on the path
 *
 * Usage:
 *   cbp2025_producers profile  <trace.gz> --pcs H2P.txt --out PROFILE.json
 *   cbp2025_producers features <trace.gz> --producers PRODUCERS.json
 *                              --commit-lag N --out PREFIX
 *
 * The output of `features` has exactly the converter's format (PREFIX.bin of
 * branch_index <i8, pc <u8, float32 columns; PREFIX.json sidecar with
 * feature_names / feature_groups), so src/dataset/raw_features.py reads it.
 *
 * ---------------------------------------------------------------------------
 * FETCH-SAFETY, the same rule as tools/cbp2025_convert.cpp:
 *
 *  - A producer value is used only from a COMMITTED instance: one at least
 *    --commit-lag instructions older than the branch. The live instance that
 *    actually decides this branch (e.g. the compare right before it) is in
 *    flight and is never read.
 *  - `inflight` counts producer instances fetched in the last commit-lag
 *    instructions. Only their PCs are used -- the front end has fetched them --
 *    never their values.
 *  - inst_hist uses only instruction TYPES on the fetched path. A type is a
 *    static property of an address (what a pre-decoder / BTB knows), not a
 *    result.
 *  - Choosing WHICH PCs are producers is offline profiling on training traces,
 *    like BranchNet's offline selection of branches; it is static metadata.
 *
 * commit-lag 1024 is the guaranteed-safe setting (ROB + fetch queue <= 1024).
 * Smaller lags are an ASSUMPTION about how far commit trails fetch and must be
 * reported as such.
 * ---------------------------------------------------------------------------
 *
 * Build:
 *   clang++ -std=c++17 -O3 -I third_party/mbplib/include \
 *     tools/cbp2025_producers.cpp -lz -o build/cbp2025_producers
 */

#include <cmath>
#include <map>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "cbp2025_trace.hpp"

using json = nlohmann::json;
using namespace cbp25;

namespace {

constexpr int64_t kWindow = 4096;   // dataflow look-back, instructions
constexpr int kMaxDepth = 3;
constexpr int kMaxProducers = 4;
constexpr int kHistory = 64;        // instances remembered per producer PC
constexpr int kTypeGroups = 6;      // alu, load, store, fp, slow_alu, branch

struct Rec {
  uint64_t pc = 0;
  uint8_t cls = 0;
  uint8_t nsrc = 0;
  int64_t srcw[3] = {-1, -1, -1};   // instr_no of the writer of each int source
};

int TypeGroup(uint8_t c) {
  switch (c) {
    case kAlu: return 0;
    case kLoad: return 1;
    case kStore: return 2;
    case kFp: return 3;
    case kSlowAlu: return 4;
    default: return 5;   // every branch class
  }
}

/* Shared per-instruction bookkeeping: the dataflow ring and last writers. */
struct Dataflow {
  std::vector<Rec> ring = std::vector<Rec>(kWindow);
  int64_t last_writer[66];
  Dataflow() { std::fill(std::begin(last_writer), std::end(last_writer), -1); }

  /* Record instruction `n` (0-based) and return its record. */
  const Rec& Push(int64_t n, const Instr& r) {
    Rec& rec = ring[n % kWindow];
    rec = Rec{};
    rec.pc = r.pc;
    rec.cls = r.cls;
    for (int q = 0; q < r.n_in && rec.nsrc < 3; ++q) {
      const uint8_t reg = r.in_regs[q];
      if (!IsIntReg(reg) || reg == 65) continue;
      rec.srcw[rec.nsrc++] = last_writer[reg];
    }
    for (int q = 0; q < r.n_out; ++q) {
      const uint8_t reg = r.out_regs[q];
      if (IsIntReg(reg) && reg != 65) last_writer[reg] = n;
    }
    return rec;
  }

  bool InWindow(int64_t n, int64_t w) const { return w >= 0 && n - w < kWindow; }
};

void Usage() {
  std::cerr << "usage:\n"
               "  cbp2025_producers profile  <trace.gz> --pcs H2P.txt --out PROFILE.json\n"
               "  cbp2025_producers features <trace.gz> --producers P.json "
               "--commit-lag N --out PREFIX\n";
}

/* ------------------------------------------------------------------ profile */

int Profile(const std::string& trace, const std::string& pcs_path,
            const std::string& out_path) {
  const auto targets = ReadPcList(pcs_path);
  // branch pc -> (depth, producer pc) -> count ; and producer pc -> class
  std::unordered_map<uint64_t, std::map<std::pair<int, uint64_t>, int64_t>> chain;
  std::unordered_map<uint64_t, int64_t> instances;
  std::unordered_map<uint64_t, uint8_t> cls_of;
  Dataflow df;
  GzReader in(trace);
  Instr r;
  int64_t n = -1;
  while (ReadInstr(in, r)) {
    ++n;
    const Rec& rec = df.Push(n, r);
    if (r.cls != kCond || !targets.count(r.pc)) continue;
    ++instances[r.pc];
    auto& counts = chain[r.pc];
    // Breadth-first walk back through writers, depth 1..kMaxDepth. A static
    // producer is counted once per branch execution.
    std::vector<int64_t> frontier(rec.srcw, rec.srcw + rec.nsrc);
    std::unordered_set<uint64_t> seen_pc;
    for (int depth = 1; depth <= kMaxDepth && !frontier.empty(); ++depth) {
      std::vector<int64_t> next;
      for (int64_t w : frontier) {
        if (!df.InWindow(n, w)) continue;
        const Rec& p = df.ring[w % kWindow];
        if (seen_pc.insert(p.pc).second) {
          ++counts[{depth, p.pc}];
          cls_of[p.pc] = p.cls;
        }
        for (int k = 0; k < p.nsrc; ++k) next.push_back(p.srcw[k]);
      }
      frontier.swap(next);
    }
  }
  json out = json::object();
  for (const auto& [bpc, counts] : chain) {
    json list = json::array();
    for (const auto& [key, c] : counts) {
      char buf[32];
      snprintf(buf, sizeof buf, "0x%llx", (unsigned long long)key.second);
      list.push_back({{"depth", key.first}, {"pc", buf}, {"count", c},
                      {"cls", cls_of[key.second]}});
    }
    char buf[32];
    snprintf(buf, sizeof buf, "0x%llx", (unsigned long long)bpc);
    out[buf] = {{"instances", instances[bpc]}, {"chain", list}};
  }
  std::ofstream(out_path) << out.dump(1) << '\n';
  return 0;
}

/* ----------------------------------------------------------------- features */

struct Inst {
  int64_t n;
  uint64_t value;
  uint64_t ea;
};

struct ProducerState {
  bool is_load = false;
  bool is_flags = false;
  std::vector<Inst> hist;      // ring of the last kHistory instances
  int head = 0;
  int size = 0;
  void Add(const Inst& x) {
    if (hist.empty()) hist.resize(kHistory);
    hist[head] = x;
    head = (head + 1) % kHistory;
    size = std::min(size + 1, kHistory);
  }
  // k-th most recent instance, k = 0 is the newest
  const Inst& Recent(int k) const {
    return hist[(head - 1 - k + 2 * kHistory) % kHistory];
  }
};

float L2(double v, double scale) {
  return static_cast<float>(std::log2(1.0 + v) / scale);
}

std::vector<std::string> FeatureNames(std::vector<std::string>* groups) {
  std::vector<std::string> n;
  for (int s = 0; s < kMaxProducers; ++s) {
    const std::string p = "prod" + std::to_string(s) + "_";
    for (const char* f : {"valid", "inflight", "age", "v_zero", "v_neg",
                          "v_logmag", "v_bit0", "v_bit1", "v_bit2", "v_bit3",
                          "d_zero", "d_neg", "d_logmag", "a_zero", "a_logmag"}) {
      n.push_back(p + f);
      groups->push_back("producers");
    }
  }
  const char* tg[kTypeGroups] = {"alu", "load", "store", "fp", "slow_alu", "branch"};
  for (int w : {16, 64})
    for (int g = 0; g < kTypeGroups; ++g) {
      n.push_back("last" + std::to_string(w) + "_" + tg[g]);
      groups->push_back("inst_hist");
    }
  for (int k = 0; k < 8; ++k)
    for (int g = 0; g < kTypeGroups; ++g) {
      n.push_back("prev" + std::to_string(k) + "_" + tg[g]);
      groups->push_back("inst_hist");
    }
  return n;
}

int Features(const std::string& trace, const std::string& producers_path,
             int64_t lag, const std::string& prefix) {
  // producers.json: {"<branch pc>": ["<producer pc>", ...], ...}
  json pj = json::parse(std::ifstream(producers_path));
  std::unordered_map<uint64_t, std::vector<uint64_t>> slots;
  std::unordered_map<uint64_t, ProducerState> prod;
  for (auto it = pj.begin(); it != pj.end(); ++it) {
    const uint64_t bpc = std::stoull(it.key(), nullptr, 16);
    auto& v = slots[bpc];
    for (const auto& p : it.value()) {
      if (v.size() >= kMaxProducers) break;
      const uint64_t ppc = std::stoull(p.get<std::string>(), nullptr, 16);
      v.push_back(ppc);
      prod[ppc];
    }
  }
  std::vector<std::string> groups;
  const auto names = FeatureNames(&groups);
  const size_t nf = names.size();
  std::vector<char> rec(16 + 4 * nf);
  std::ofstream out(prefix + ".bin", std::ios::binary);
  if (!out) { std::cerr << "cannot write " << prefix << ".bin\n"; return 1; }

  std::vector<uint8_t> types(64, 255);   // ring of the last 64 type groups
  int64_t branch_idx = 0, records = 0;
  GzReader in(trace);
  Instr r;
  int64_t n = -1;
  while (ReadInstr(in, r)) {
    ++n;
    if (IsBranch(r.cls)) {
      auto sit = slots.find(r.pc);
      if (r.cls == kCond && sit != slots.end()) {
        float* f = reinterpret_cast<float*>(rec.data() + 16);
        size_t k = 0;
        for (int s = 0; s < kMaxProducers; ++s) {
          float row[15] = {0};
          if (s < static_cast<int>(sit->second.size())) {
            const ProducerState& ps = prod[sit->second[s]];
            int inflight = 0, c1 = -1, c2 = -1;
            for (int j = 0; j < ps.size; ++j) {
              const Inst& x = ps.Recent(j);
              if (n - x.n < lag) { ++inflight; continue; }   // not committed
              if (c1 < 0) c1 = j; else { c2 = j; break; }
            }
            row[1] = L2(inflight, 6.0);
            if (c1 >= 0) {
              const Inst& a = ps.Recent(c1);
              const uint64_t v = a.value;
              row[0] = 1;
              row[2] = L2(static_cast<double>(n - a.n), 32.0);
              row[3] = v == 0;
              row[4] = !ps.is_flags && (v >> 63);
              const double mag = static_cast<double>(
                  (v >> 63) && !ps.is_flags ? ~v + 1 : v);
              row[5] = ps.is_flags ? 0.0f : L2(mag, 64.0);
              const uint64_t nib = ps.is_flags ? (v >> 28) & 0xF : v & 0xF;
              for (int b = 0; b < 4; ++b) row[6 + b] = (nib >> b) & 1u;
              if (c2 >= 0) {
                const Inst& bq = ps.Recent(c2);
                const int64_t d = static_cast<int64_t>(v - bq.value);
                row[10] = d == 0;
                row[11] = d < 0;
                row[12] = L2(std::fabs(static_cast<double>(d)), 64.0);
                if (ps.is_load) {
                  const int64_t st = static_cast<int64_t>(a.ea - bq.ea);
                  row[13] = st == 0;
                  row[14] = L2(std::fabs(static_cast<double>(st)), 32.0);
                }
              }
            }
          }
          for (float x : row) f[k++] = x;
        }
        // inst_hist: types of the instructions BEFORE this branch.
        for (int w : {16, 64}) {
          int cnt[kTypeGroups] = {0}, tot = 0;
          for (int j = 1; j <= w && j <= n; ++j) {
            const uint8_t t = types[(n - j) % 64];
            if (t < kTypeGroups) { ++cnt[t]; ++tot; }
          }
          for (int g = 0; g < kTypeGroups; ++g)
            f[k++] = tot ? static_cast<float>(cnt[g]) / tot : 0.0f;
        }
        for (int j = 1; j <= 8; ++j) {
          const uint8_t t = j <= n ? types[(n - j) % 64] : 255;
          for (int g = 0; g < kTypeGroups; ++g) f[k++] = (t == g) ? 1.0f : 0.0f;
        }
        if (k != nf) { std::cerr << "feature count mismatch\n"; return 1; }
        std::memcpy(rec.data(), &branch_idx, 8);
        std::memcpy(rec.data() + 8, &r.pc, 8);
        out.write(rec.data(), static_cast<std::streamsize>(rec.size()));
        ++records;
      }
      ++branch_idx;
    }
    // After the branch's own features: record this instruction.
    types[n % 64] = static_cast<uint8_t>(TypeGroup(r.cls));
    auto pit = prod.find(r.pc);
    if (pit != prod.end()) {
      uint64_t value = 0;
      for (int q = 0; q < r.n_out; ++q) {
        if (IsIntReg(r.out_regs[q]) && r.out_regs[q] != 65) {
          value = r.out_int_values[q];
          pit->second.is_flags = r.out_regs[q] == 64;
          break;
        }
      }
      pit->second.is_load = r.cls == kLoad;
      pit->second.Add({n, value, r.ea});
    }
  }
  out.close();
  json sidecar = {
      {"format", "cbp2025-fetch-safe-features"}, {"version", 1},
      {"trace", trace}, {"records", records}, {"record_bytes", rec.size()},
      {"feature_names", names}, {"feature_groups", groups},
      {"feature_dtype", "<f4"}, {"commit_lag", lag},
      {"producers_file", producers_path},
      {"note", "producers: committed history of each branch's dataflow "
               "producers (instances >= commit_lag instructions old); "
               "inflight counts fetched-not-committed instances by PC only. "
               "inst_hist: instruction types on the fetched path."}};
  std::ofstream(prefix + ".json") << sidecar.dump(2) << '\n';
  std::cerr << "wrote " << records << " records\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { Usage(); return 2; }
  const std::string mode = argv[1], trace = argv[2];
  std::string pcs, out, producers;
  int64_t lag = 1024;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::cerr << a << " needs a value\n"; exit(2); }
      return argv[++i];
    };
    if (a == "--pcs") pcs = next();
    else if (a == "--out") out = next();
    else if (a == "--producers") producers = next();
    else if (a == "--commit-lag") lag = std::stoll(next());
    else { Usage(); return 2; }
  }
  if (mode == "profile" && !pcs.empty() && !out.empty())
    return Profile(trace, pcs, out);
  if (mode == "features" && !producers.empty() && !out.empty())
    return Features(trace, producers, lag, out);
  Usage();
  return 2;
}
