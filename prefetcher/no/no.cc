#include "no.h"
#include <fstream>
#include <string>
#include <iostream>  // for std::cout

std::string access_type_to_string(access_type type) {
  switch (type) {
    case access_type::LOAD:        return "LOAD";
    case access_type::RFO:         return "RFO";
    case access_type::PREFETCH:    return "PREFETCH";
    case access_type::WRITE:       return "WRITE";
    case access_type::TRANSLATION: return "TRANSLATION";
    default:                       return "UNKNOWN";
  }
}

uint32_t no::prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                      uint8_t cache_hit, bool useful_prefetch,
                                      access_type type, uint32_t metadata_in)
{
  static constexpr size_t SKIP_FIRST_N = 100'000'000;  // Approximate warmup
  static size_t op_counter = 0;
  static bool initialized = false;
  static bool notified = false;

  op_counter++;
  if (op_counter < SKIP_FIRST_N)
    return metadata_in;

  std::ofstream debug_file;
  std::ofstream miss_file;

  if (!initialized) {
    debug_file.open("debug.csv", std::ios_base::out);  // Overwrite first time
    debug_file << "addr,ip,cache_hit,type\n";           // CSV header
    miss_file.open("miss.csv", std::ios_base::out);
    miss_file << "addr,ip,cache_hit,type\n";           // CSV header for misses
    initialized = true;
    if (!notified) {
      std::cout << "[LOG] Warmup complete, CSV logging started to debug.csv\n";
      notified = true;
    }
  } else {
    debug_file.open("debug.csv", std::ios_base::app);  // Append after first
    miss_file.open("miss.csv", std::ios_base::app);
  }

  debug_file << std::hex << addr << ","
             << ip << ","
             << std::dec << static_cast<int>(cache_hit) << ","
             << access_type_to_string(type) << "\n";

  if (cache_hit == 0) {  // Only log misses
    miss_file << std::hex << addr << ","
              << ip << ","
              << std::dec << static_cast<int>(cache_hit) << ","
              << access_type_to_string(type) << "\n";
  }
  return metadata_in;
}

uint32_t no::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}
