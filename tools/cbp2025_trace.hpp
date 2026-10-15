/* CBP2025 instruction-trace reader, shared by the converter and the
 * producer-feature tool so the two can never decode the format differently.
 *
 * Layout: the CBP2025 framework's lib/trace_reader.h (readInstr). Registers:
 * 0-30 GPR, 31 SP, 32-63 SIMD, 64 NZCV flags, 65 zero register.
 */
#ifndef CBP2025_TRACE_HPP_
#define CBP2025_TRACE_HPP_

#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace cbp25 {

enum Cls : uint8_t {
  kAlu = 0, kLoad = 1, kStore = 2, kCond = 3, kUncondDirect = 4,
  kUncondIndirect = 5, kFp = 6, kSlowAlu = 7, kUndef = 8, kCallDirect = 9,
  kCallIndirect = 10, kReturn = 11,
};

inline bool IsBranch(uint8_t c) {
  return c == kCond || c == kUncondDirect || c == kUncondIndirect ||
         c == kCallDirect || c == kCallIndirect || c == kReturn;
}
inline bool IsIntReg(uint8_t r) { return r < 32 || r == 64 || r == 65; }

/* mbp::Branch::OpCode for each CBP2025 branch class. Returns are RET|IND,
 * matching what the CBP-5 SBBT traces use (opcode 6). */
inline uint8_t SbbtOpcode(uint8_t c) {
  switch (c) {
    case kCond: return 0b0001;            // CND | JUMP
    case kUncondDirect: return 0b0000;    // JUMP
    case kUncondIndirect: return 0b0010;  // IND | JUMP
    case kCallDirect: return 0b1000;      // CALL
    case kCallIndirect: return 0b1010;    // CALL | IND
    case kReturn: return 0b0110;          // RET | IND
  }
  return 0;
}

struct Instr {
  uint64_t pc = 0;
  uint8_t cls = 0;
  bool taken = false;
  uint64_t target = 0;
  uint64_t ea = 0;          // load/store effective address
  uint8_t n_in = 0;
  uint8_t in_regs[256];
  uint8_t n_out = 0;
  uint8_t out_regs[256];
  uint64_t out_int_values[256];  // only meaningful where IsIntReg(out_regs[i])
};

/* Buffered gzip reader with exact-size reads. */
class GzReader {
 public:
  explicit GzReader(const std::string& path) {
    fh_ = gzopen(path.c_str(), "rb");
    if (!fh_) { std::cerr << "cannot open " << path << '\n'; exit(1); }
    gzbuffer(fh_, 1 << 20);
    buf_.resize(1 << 22);
  }
  ~GzReader() { if (fh_) gzclose(fh_); }

  /* false only at a clean end of stream before the first byte. */
  bool Read(void* dst, size_t n, bool allow_eof = false) {
    auto* out = static_cast<uint8_t*>(dst);
    while (n > 0) {
      if (pos_ == end_) {
        int got = gzread(fh_, buf_.data(), static_cast<unsigned>(buf_.size()));
        if (got < 0) { std::cerr << "gzip error\n"; exit(1); }
        if (got == 0) {
          if (allow_eof) return false;
          std::cerr << "truncated trace record\n";
          exit(1);
        }
        pos_ = 0; end_ = static_cast<size_t>(got);
      }
      size_t take = std::min(n, end_ - pos_);
      std::memcpy(out, buf_.data() + pos_, take);
      pos_ += take; out += take; n -= take;
      allow_eof = false;  // a partial record is truncation, not EOF
    }
    return true;
  }

 private:
  gzFile fh_ = nullptr;
  std::vector<uint8_t> buf_;
  size_t pos_ = 0, end_ = 0;
};

inline bool ReadInstr(GzReader& in, Instr& r) {
  if (!in.Read(&r.pc, 8, /*allow_eof=*/true)) return false;
  in.Read(&r.cls, 1);
  if (r.cls == kUndef || r.cls > kReturn) {
    std::cerr << "bad instruction class " << int(r.cls) << " at pc 0x"
              << std::hex << r.pc << std::dec << '\n';
    exit(1);
  }
  if (r.cls == kLoad || r.cls == kStore) {
    uint8_t size, base_upd;
    in.Read(&r.ea, 8); in.Read(&size, 1); in.Read(&base_upd, 1);
    if (r.cls == kStore) { uint8_t reg_off; in.Read(&reg_off, 1); }
  }
  r.taken = false;
  r.target = r.pc + 4;
  if (IsBranch(r.cls)) {
    uint8_t t; in.Read(&t, 1);
    r.taken = t != 0;
    if (r.taken) in.Read(&r.target, 8);
  }
  in.Read(&r.n_in, 1);
  in.Read(r.in_regs, r.n_in);
  in.Read(&r.n_out, 1);
  in.Read(r.out_regs, r.n_out);
  for (int i = 0; i < r.n_out; ++i) {
    if (IsIntReg(r.out_regs[i])) {
      in.Read(&r.out_int_values[i], 8);
    } else {
      uint64_t simd[2]; in.Read(simd, 16);
      r.out_int_values[i] = 0;
    }
  }
  return true;
}

inline std::unordered_set<uint64_t> ReadPcList(const std::string& path) {
  std::ifstream fh(path);
  if (!fh) { std::cerr << "cannot read " << path << '\n'; exit(2); }
  std::unordered_set<uint64_t> pcs;
  std::string line;
  while (std::getline(fh, line)) {
    if (auto h = line.find('#'); h != std::string::npos) line.resize(h);
    auto b = line.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) continue;
    auto e = line.find_last_not_of(" \t\r\n");
    pcs.insert(std::stoull(line.substr(b, e - b + 1), nullptr, 16));
  }
  return pcs;
}

inline constexpr uint64_t kIpMask = (uint64_t{1} << 52) - 1;

}  // namespace cbp25

#endif  // CBP2025_TRACE_HPP_
