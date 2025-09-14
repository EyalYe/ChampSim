#ifndef PREFETCHER_BNNPREF_H
#define PREFETCHER_BNNPREF_H

#include <cstdint>

#include "champsim.h"
#include "modules.h"

#include <deque>
#include <unordered_map>
#include <utility>
#include <sstream>
#include <cstdio>
#include <cstdlib>

class bnnpref: public champsim::modules::prefetcher {
public:
  using prefetcher::prefetcher;

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);

private:
  static constexpr int HISTORY_LENGTH = 64;
  static constexpr const char* BASE_DIR = "prefetcher/bnnpref/model/files/tmp";
  static constexpr const char* SCRIPT_PATH = "prefetcher/bnnpref/model/infer.py";
  static constexpr const char* INFER_SOCK_PATH = "/tmp/bnnpref.sock";
  static constexpr int PREF_TOPK = 10;
  static constexpr int BLOCK_BYTES = 64;
  static constexpr int CNNPREF_WARMUP_DEMANDS = 200000000 ;
};

#endif

