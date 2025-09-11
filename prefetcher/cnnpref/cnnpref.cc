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
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string>

// Send an already-built JSON payload to the UNIX socket, return server JSON (optional)
bool send_infer_request_raw(const std::string& sock, const std::string& json_payload, std::string* resp_out) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return false;
  sockaddr_un addr{}; addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock.c_str());
  if (connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0) { close(fd); return false; }

  if (write(fd, json_payload.data(), json_payload.size()) < 0) { close(fd); return false; }
  shutdown(fd, SHUT_WR);

  // read response
  char buf[4096]; std::string resp;
  ssize_t n;
  while ((n = read(fd, buf, sizeof(buf))) > 0) resp.append(buf, buf+n);
  close(fd);
  if (resp_out) *resp_out = std::move(resp);
  return true;
}

bool send_infer_request(const std::string& sock, const std::string& in_csv, const std::string& out_txt, int topk=10, int hist=64, int block_bytes=64) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return false;
  sockaddr_un addr{}; addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock.c_str());
  if (connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0) { close(fd); return false; }

  std::string payload = std::string("{\"in\":\"") + in_csv + "\",\"out\":\"" + out_txt +
                        "\",\"topk\":" + std::to_string(topk) +
                        ",\"hist\":" + std::to_string(hist) +
                        ",\"block_bytes\":" + std::to_string(block_bytes) + "}";
  if (write(fd, payload.data(), payload.size()) < 0) { close(fd); return false; }
  shutdown(fd, SHUT_WR);

  // read response (optional)
  char buf[4096]; std::string resp;
  ssize_t n;
  while ((n = read(fd, buf, sizeof(buf))) > 0) resp.append(buf, buf+n);
  close(fd);
  // you can parse resp JSON if you want, or ignore
  return true;
}


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

  // Skip prefetching during warmup
  // ── Warmup gate: count only demand accesses, skip prefetching during warmup ──
  /*
  static size_t demand_accesses = 0;
  demand_accesses += 1;

  if (demand_accesses < CNNPREF_WARMUP_DEMANDS) {
    if (demand_accesses == 1) {
      std::cerr << "cnnpref: warming up, skipping prefetches until "
                << CNNPREF_WARMUP_DEMANDS << " demand accesses\n";
    }
    else if (demand_accesses ==  CNNPREF_WARMUP_DEMANDS - 1) {
      std::cerr << "cnnpref: warmup done, starting prefetching\n";
    } else if (demand_accesses == 10000000) {
      std::cerr << "cnnpref: still warming up, "
                << (CNNPREF_WARMUP_DEMANDS - demand_accesses)
                << " more demand accesses to go\n";
    }
    return metadata_in;
  }
  */
  /*
  */
  // ── Unique per-call files ───────────────────────────────────────────────
  static std::atomic<uint64_t> seq{0};
  const uint64_t id = seq.fetch_add(1, std::memory_order_relaxed);

  const std::string base_dir = BASE_DIR;
  std::error_code ec;
  std::filesystem::create_directories(base_dir, ec);

  const std::string out_path  = base_dir + "/out_"  + std::to_string(id) + ".txt";
  /*
  const std::string hist_path = base_dir + "/hist_" + std::to_string(id) + ".csv";
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

  // This blocks until the server writes `out_path`, so it's safe to read it right after.
  bool ok = send_infer_request(INFER_SOCK_PATH, hist_path, out_path, PREF_TOPK, HISTORY_LENGTH, BLOCK_BYTES);
  if (!ok) {
    std::cerr << "cnnpref: infer server request failed (socket " << INFER_SOCK_PATH << ")\n";
    std::filesystem::remove(hist_path, ec);
    return metadata_in;
  }
  
  */
  // Build inline JSON with the last HISTORY_LENGTH entries (addr, ip, hit)
  // We send hex strings (0x...) so Python can parse with int(x, 0)
  std::ostringstream req;
  req << "{\"hist_inline\":{";

  req << "\"addr\":[";
  for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
    if (i) req << ",";
    req << "\"" << std::hex << access_history[i] << "\"";
  }
  req << "],";

  req << "\"ip\":[";
  for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
    if (i) req << ",";
    req << "\"" << std::hex << ip_history[i] << "\"";
  }
  req << "],";

  req << "\"hit\":[";
  for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
    if (i) req << ",";
    req << std::dec << static_cast<unsigned>(cache_hit_history[i]);
  }
  req << "]},";
  req << "\"out\":\"" << out_path << "\",";
  req << "\"topk\":" << PREF_TOPK << ",";
  req << "\"hist\":" << HISTORY_LENGTH << ",";
  req << "\"block_bytes\":" << BLOCK_BYTES;
  req << "}";

  std::string server_resp;
  bool ok = send_infer_request_raw(INFER_SOCK_PATH, req.str(), &server_resp);
  if (!ok) {
    std::cerr << "cnnpref: infer server request failed (socket " << INFER_SOCK_PATH << ")\n";
    std::error_code ec2; std::filesystem::remove(out_path, ec2);
    return metadata_in;
  }
  // (optional) you can inspect server_resp JSON for diagnostics if you want


  // Read predictions (one hex per line or a single token)
  {
    std::ifstream fin(out_path);
    if (!fin) {
      std::cerr << "cnnpref: cannot open " << out_path << "\n";
      //std::filesystem::remove(hist_path, ec);
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

  //std::filesystem::remove(hist_path, ec);
  std::filesystem::remove(out_path, ec);
  return metadata_in;
}

uint32_t cnnpref::prefetcher_cache_fill(champsim::address addr, long set, long way,
                                        uint8_t prefetch, champsim::address evicted_addr,
                                        uint32_t metadata_in)
{
  return metadata_in;
}

