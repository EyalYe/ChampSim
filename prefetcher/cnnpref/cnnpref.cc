// cnnpref.cc (relevant parts)
#include "cnnpref.h"
#include <deque>
#include <atomic>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>

#ifndef HISTORY_LENGTH
#define HISTORY_LENGTH 64
#endif

static std::string access_type_to_string(access_type type) {
  switch (type) {
    case access_type::LOAD:        return "LOAD";
    case access_type::RFO:         return "RFO";
    case access_type::PREFETCH:    return "PREFETCH";
    case access_type::WRITE:       return "WRITE";
    case access_type::TRANSLATION: return "TRANSLATION";
    default:                       return "UNKNOWN";
  }
}

uint32_t cnnpref::prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                           uint8_t cache_hit, bool useful_prefetch,
                                           access_type type, uint32_t metadata_in)
{
  static std::deque<champsim::address> access_history;
  static std::deque<champsim::address> ip_history;
  static std::deque<uint8_t>           cache_hit_history;
  static std::deque<std::string>       type_history;

  if (access_history.empty()) {
    access_history.resize(HISTORY_LENGTH);
    ip_history.resize(HISTORY_LENGTH);
    cache_hit_history.resize(HISTORY_LENGTH, 0u);
    type_history.resize(HISTORY_LENGTH, "NONE");
  }

  access_history.pop_front();      access_history.push_back(addr);
  ip_history.pop_front();          ip_history.push_back(ip);
  cache_hit_history.pop_front();   cache_hit_history.push_back(cache_hit);
  type_history.pop_front();        type_history.push_back(access_type_to_string(type));

  // ── Unique per-call files ───────────────────────────────────────────────
  static std::atomic<uint64_t> seq{0};
  const uint64_t id = seq.fetch_add(1, std::memory_order_relaxed);

  const std::string base_dir = BASE_DIR;
  std::error_code ec;
  std::filesystem::create_directories(base_dir, ec);

  const std::string hist_path = base_dir + "/hist_" + std::to_string(id) + ".csv";
  const std::string out_path  = base_dir + "/out_"  + std::to_string(id) + ".txt";

  // Build the CSV in-memory to avoid partial lines
  std::ostringstream oss;
  for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
    oss << std::showbase << std::hex
        << access_history[i] << ","
        << ip_history[i]     << ","
        << std::dec << int(cache_hit_history[i]) << ","
        << type_history[i]   << "\n";
  }

  {
    std::ofstream history_file(hist_path, std::ios::out | std::ios::trunc);
    if (!history_file) {
      std::cerr << "cnnpref: failed to open " << hist_path << "\n";
      return metadata_in;
    }
    history_file << oss.str();
    history_file.flush();
    if (!history_file) {
      std::cerr << "cnnpref: write failed for " << hist_path << "\n";
      return metadata_in;
    }
  }

  // Optional: throttle/serialize Python calls
  static std::mutex io_mutex;
  std::lock_guard<std::mutex> lk(io_mutex);


  std::string cmd = std::string("python3 \"") + SCRIPT_PATH +
                    "\" --input_file \"" + hist_path +
                    "\" --output_file \"" + out_path + "\"";

  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "cnnpref: inference script failed, code " << ret << "\n";
    std::filesystem::remove(hist_path, ec);
    return metadata_in;
  }

  // Read predictions (one hex per line or a single token)
  {
    std::ifstream fin(out_path);
    if (!fin) {
      std::cerr << "cnnpref: cannot open " << out_path << "\n";
      std::filesystem::remove(hist_path, ec);
      std::filesystem::remove(out_path, ec);
      return metadata_in;
    }
    std::string token;
    while (fin >> token) {
      try {
        // base=0 lets "0x..." work
        unsigned long long raw = std::stoull(token, nullptr, 0);
        prefetch_line(champsim::address{raw}, /*fill_this_level=*/true, metadata_in);
      } catch (...) {
        // ignore malformed tokens
        std::cerr << "cnnpref: cannot parse token '" << token << "' in " << out_path << "\n";
      }
    }
  }

  std::filesystem::remove(hist_path, ec);
  std::filesystem::remove(out_path, ec);
  return metadata_in;
}

uint32_t cnnpref::prefetcher_cache_fill(champsim::address addr, long set, long way,
                                        uint8_t prefetch, champsim::address evicted_addr,
                                        uint32_t metadata_in)
{
  return metadata_in;
}

