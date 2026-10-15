// Branch-only replay of CBP2016 TAGE-SC-L 64KB (third_party/cbp2016_tage_sc_l/cbp2016_tage_sc_l.h)
// over a CBP2025 instruction trace (.gz).  NOT the CBP2025 OoO simulator: there is no
// pipeline model, so predict -> history_update -> update happen back-to-back per branch.
//
// Decoder follows the CBP2025 framework's lib/trace_reader.h (layout in the CBP2025 brief).
// Emits: aggregate JSON (stdout), per-PC CSV, and a per-conditional-branch log
// (ordinal among conditional branches, so it joins with FeatureExtractor sample rows
// when sample_stride == 1 and warmup == 0): byte0 = mispredicted, byte1 = predicted dir,
// byte2 = HitBank (clipped 255), byte3 = HighConf | LowConf<<1 | (AltBank!=0)<<2.
//
// Build (see results/tage/BUILD.md):
//   clang++ -std=c++17 -O3 -I<cbp2025> tools/tage_cbp2025.cpp -lz -o build/tage_cbp2025
#include <zlib.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include "cbp2016_tage_sc_l.h"

static gzFile g;
static bool rd(void* p, size_t n) { return gzread(g, p, (unsigned)n) == (int)n; }
static uint64_t seq = 0;
// Register-dependence stats for scored conditional branches (lag study, see results/REPORT).
static uint64_t ST_flags_only = 0, ST_nin[6] = {0};
static const uint64_t LAGS[4] = {16, 64, 256, 1024};
static uint64_t ST_fresh[4] = {0}, ST_fresh_miss[4] = {0}, ST_haveprod = 0, ST_haveprod_miss = 0;   // persists across traces so warm-up traces share one predictor
struct S { uint64_t n = 0, t = 0, m = 0; };

// Replays one trace through the (persistent) predictor. If !scored, only trains.
static int run_trace(const char* path, bool scored, uint64_t limit, FILE* lf, uint64_t& instr, uint64_t& nbr,
                     uint64_t& ncond, uint64_t& miss, uint64_t* cls_hist, std::unordered_map<uint64_t, S>& per) {
    g = gzopen(path, "rb");
    if (!g) { perror("gzopen"); return 1; }
    gzbuffer(g, 1 << 20);
    instr = nbr = ncond = miss = 0;
    int64_t last_w[66]; for (int i = 0; i < 66; i++) last_w[i] = -1;
    for (;;) {
        uint64_t pc;
        if (gzread(g, &pc, 8) != 8) break;
        uint8_t cls; if (!rd(&cls, 1)) return 3;
        if (cls > 11 || cls == 8) { fprintf(stderr, "bad class %d at %llu\n", cls, (unsigned long long)instr); return 3; }
        if (cls == 1 || cls == 2) {
            uint64_t ea; uint8_t sz, bu, ro;
            rd(&ea, 8); rd(&sz, 1); rd(&bu, 1);
            if (cls == 2) rd(&ro, 1);
        }
        bool isbr = (cls == 3 || cls == 4 || cls == 5 || cls == 9 || cls == 10 || cls == 11);
        bool taken = false; uint64_t next = pc + 4;
        if (isbr) {
            uint8_t t; rd(&t, 1); taken = t;
            if (taken) { rd(&next, 8); }
        }
        uint8_t nin; rd(&nin, 1);
        uint8_t regs[256]; if (nin) rd(regs, nin);
        uint8_t nout; rd(&nout, 1);
        uint8_t outr[256]; if (nout) rd(outr, nout);
        for (int i = 0; i < nout; i++) {
            uint64_t v[2];
            rd(&v[0], 8);
            // SIMD regs (32..63) carry 16 bytes; 0-31 and 64,65 are 8 bytes (trace_reader.h reg_is_int)
            if (outr[i] >= 32 && outr[i] < 64) rd(&v[1], 8);
        }
        if (cls == 3 && scored) {
            ST_nin[nin > 5 ? 5 : nin]++;
            if (nin == 1 && regs[0] == 64) ST_flags_only++;
        }
        int64_t nearest = -1;
        for (int i = 0; i < nin; i++) { if (regs[i] < 66 && last_w[regs[i]] >= 0) { int64_t d = (int64_t)instr - last_w[regs[i]]; if (nearest < 0 || d < nearest) nearest = d; } }
        cls_hist[cls]++;
        const uint64_t this_instr = instr;
        instr++;
        if (isbr) {
            nbr++;
            if (cls == 3) {
                bool pred = cbp2016_tage_sc_l.predict(seq, 0, pc);
                uint8_t hb = (uint8_t)std::min(cbp2016_tage_sc_l.HitBank, 255);
                uint8_t cf = (cbp2016_tage_sc_l.HighConf ? 1 : 0) | (cbp2016_tage_sc_l.LowConf ? 2 : 0) | (cbp2016_tage_sc_l.AltBank != 0 ? 4 : 0);
                cbp2016_tage_sc_l.history_update(seq, 0, pc, 1, pred, taken, next);
                cbp2016_tage_sc_l.update(seq, 0, pc, taken, pred, next);
                ncond++;
                bool m = (pred != taken);
                miss += m;
                if (scored && nearest >= 0) { ST_haveprod++; ST_haveprod_miss += m; for (int l = 0; l < 4; l++) if ((uint64_t)nearest >= LAGS[l]) { ST_fresh[l]++; ST_fresh_miss[l] += m; } }
                if (!scored) { miss -= m; ncond--; }
                else { auto& s = per[pc]; s.n++; s.t += taken; s.m += m; }
                if (lf && scored) { uint8_t b[4] = {(uint8_t)m, (uint8_t)pred, hb, cf}; fwrite(b, 1, 4, lf); }
                seq++;
            } else {
                int bt = (cls == 4 || cls == 9) ? 0 : 2;
                cbp2016_tage_sc_l.TrackOtherInst(pc, bt, true, true, next);
            }
        }
        for (int i = 0; i < nout; i++) if (outr[i] < 66) last_w[outr[i]] = (int64_t)this_instr;
        if (limit && instr >= limit) break;
    }
    gzclose(g);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s trace.gz [--per-pc f.csv] [--log f.bin] [--limit N] [--warm a.gz,b.gz]\n", argv[0]); return 2; }
    std::string per_pc, logf, warm; uint64_t limit = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--per-pc") && i + 1 < argc) per_pc = argv[++i];
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) logf = argv[++i];
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc) warm = argv[++i];
    }
    FILE* lf = logf.empty() ? nullptr : fopen(logf.c_str(), "wb");
    uint64_t instr = 0, nbr = 0, ncond = 0, miss = 0;
    uint64_t cls_hist[12] = {0};
    std::unordered_map<uint64_t, S> per;
    { // warm-up traces: trained on, never scored
        size_t a = 0;
        while (a < warm.size()) {
            size_t b = warm.find(',', a); if (b == std::string::npos) b = warm.size();
            std::string w = warm.substr(a, b - a); a = b + 1;
            if (w.empty()) continue;
            std::unordered_map<uint64_t, S> dummy; uint64_t ch[12] = {0};
            if (run_trace(w.c_str(), false, 0, nullptr, instr, nbr, ncond, miss, ch, dummy)) return 1;
        }
    }
    if (run_trace(argv[1], true, limit, lf, instr, nbr, ncond, miss, cls_hist, per)) return 1;
    if (lf) fclose(lf);
    if (!per_pc.empty()) {
        FILE* f = fopen(per_pc.c_str(), "w");
        fprintf(f, "pc,n,taken,mispred\n");
        std::map<uint64_t, S> sorted(per.begin(), per.end());
        for (auto& kv : sorted) fprintf(f, "0x%llx,%llu,%llu,%llu\n", (unsigned long long)kv.first, (unsigned long long)kv.second.n, (unsigned long long)kv.second.t, (unsigned long long)kv.second.m);
        fclose(f);
    }
    printf("{\"trace\":\"%s\",\"instructions\":%llu,\"branches\":%llu,\"cond_branches\":%llu,\"mispredictions\":%llu,\"mpki\":%.4f,\"distinct_cond_pcs\":%zu,\"class_hist\":[",
           argv[1], (unsigned long long)instr, (unsigned long long)nbr, (unsigned long long)ncond, (unsigned long long)miss,
           instr ? 1000.0 * miss / instr : 0.0, per.size());
    for (int i = 0; i < 12; i++) printf("%s%llu", i ? "," : "", (unsigned long long)cls_hist[i]);
    printf("],\"cond_flags_only\":%llu,\"cond_nin_hist\":[", (unsigned long long)ST_flags_only);
    for (int i = 0; i < 6; i++) printf("%s%llu", i ? "," : "", (unsigned long long)ST_nin[i]);
    printf("],\"cond_with_producer\":%llu,\"cond_with_producer_miss\":%llu,\"fresh_lags\":[16,64,256,1024],\"fresh_cond\":[", (unsigned long long)ST_haveprod, (unsigned long long)ST_haveprod_miss);
    for (int i = 0; i < 4; i++) printf("%s%llu", i ? "," : "", (unsigned long long)ST_fresh[i]);
    printf("],\"fresh_cond_miss\":[");
    for (int i = 0; i < 4; i++) printf("%s%llu", i ? "," : "", (unsigned long long)ST_fresh_miss[i]);
    printf("]}\n");
    return 0;
}
