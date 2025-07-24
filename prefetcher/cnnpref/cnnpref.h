#ifndef PREFETCHER_CNNPREF_H
#define PREFETCHER_CNNPREF_H

#include <cstdint>

#include "champsim.h"
#include "modules.h"

#include <deque>
#include <unordered_map>
#include <utility>
#include <sstream>
#include <cstdio>
#include <cstdlib>

class cnnpref : public champsim::modules::prefetcher {
public:
  using prefetcher::prefetcher;

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);

private:
  static constexpr int HISTORY_LENGTH = 16;
  std::deque<std::pair<uint64_t, int64_t>> history; // <pc, delta>
  int64_t last_block = -1;
};

#endif
