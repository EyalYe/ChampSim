// bnnpref.cc — ChampSim prefetcher with persistent newline-framed UNIX socket RPC
#include "bnnpref.h"

#include "champsim.h"
#include "modules.h"

#include <deque>
#include <atomic>
#include <mutex>
#include <filesystem>
#include <sstream>
#include <iostream>
#include <regex>
#include <string>
#include <csignal>
#include <cerrno>

// POSIX sockets
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace std;

namespace {
// ────────────────────────── Persistent socket state ──────────────────────────
static std::mutex g_sock_mtx;
static int g_sock_fd = -1;
static std::string g_read_buf; // leftover bytes after newline for next read

// Avoid SIGPIPE when server disappears
struct SigpipeIgnorer {
  SigpipeIgnorer() { std::signal(SIGPIPE, SIG_IGN); }
} g_sigpipe_ignorer;

// If you want the server to print a progress bar, you can tune these:
static constexpr int    PROGRESS_TOTAL = 100;        // “hard” total that maps to 100%
static constexpr char   PROGRESS_LABEL[] = "bnnpref"; // shown before the bar
static constexpr bool   PROGRESS_BAR = true;         // enable/disable server-side bar

bool connect_persistent(const std::string& sock_path) {
  if (g_sock_fd != -1) return true;
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return false;

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock_path.c_str());
  if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0) {
    ::close(fd);
    return false;
  }

  // NOTE: no recv timeout (blocking). If you prefer a finite timeout, uncomment:
  // timeval tv{2, 0}; // 2 seconds
  // ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  g_sock_fd = fd;
  g_read_buf.clear();
  return true;
}

bool write_all(int fd, const char* p, size_t n) {
  while (n) {
    ssize_t w = ::send(fd, p, n,
#ifdef MSG_NOSIGNAL
                       MSG_NOSIGNAL
#else
                       0
#endif
    );
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += w; n -= size_t(w);
  }
  return true;
}

// Read one '\n'-terminated JSON response, buffering any extra bytes.
bool read_line(int fd, std::string& out) {
  for (;;) {
    // Check buffer for newline first
    auto pos = g_read_buf.find('\n');
    if (pos != std::string::npos) {
      out.assign(g_read_buf.data(), pos);
      g_read_buf.erase(0, pos + 1);
      return true;
    }
    // Need more data
    char buf[4096];
    ssize_t r = ::recv(fd, buf, sizeof(buf), 0);
    if (r == 0) return false;            // peer closed
    if (r < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue; // soft timeout → retry
      return false;
    }
    g_read_buf.append(buf, size_t(r));
    if (g_read_buf.size() > (1u << 20)) // safety cap 1MB
      return false;
  }
}

bool request_response_json_line(const std::string& sock_path,
                                const std::string& json_line,
                                std::string* resp_out) {
  if (!connect_persistent(sock_path)) {
    return false;
  }
  std::string payload = json_line;
  payload.push_back('\n');
  if (!write_all(g_sock_fd, payload.data(), payload.size())) {
    ::close(g_sock_fd); g_sock_fd = -1;
    if (!connect_persistent(sock_path)) return false;
    if (!write_all(g_sock_fd, payload.data(), payload.size())) {
      ::close(g_sock_fd); g_sock_fd = -1;
      return false;
    }
  }
  std::string resp;
  if (!read_line(g_sock_fd, resp)) {
    ::close(g_sock_fd); g_sock_fd = -1;
    return false;
  }
  if (resp_out) *resp_out = std::move(resp);
  return true;
}

// ────────────────────────── Helper ──────────────────────────
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

} // namespace

uint32_t bnnpref::prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                           uint8_t cache_hit, bool /*useful_prefetch*/,
                                           access_type type, uint32_t metadata_in)
{
  static constexpr size_t SKIP_FIRST_N = 6900000;  // Approximate warmup
  static size_t op_counter = 0;

  op_counter++;
  if (op_counter < SKIP_FIRST_N){
    if (op_counter % 100000 == 0)
      std::cout << "bnnpref: skipping first " << SKIP_FIRST_N << " ops, now at " << op_counter << "\n";
    return metadata_in;

  }
  // ── Sliding history buffers (kept static across calls) ──
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

  // ── Build inline JSON with last HISTORY_LENGTH entries + progress options ──
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
  req << "\"topk\":" << PREF_TOPK << ",";
  req << "\"hist\":" << HISTORY_LENGTH << ",";
  req << "\"block_bytes\":" << BLOCK_BYTES << ",";
  // Server-side progress bar controls (printed in server stdout)
  req << "\"progress_bar\":" << (PROGRESS_BAR ? "true" : "false") << ",";
  req << "\"progress_total\":" << PROGRESS_TOTAL << ",";
  req << "\"progress_label\":\"" << PROGRESS_LABEL << "\"";
  req << "}"; // end root JSON

  // ── Send via persistent socket and parse inline predictions ──
  std::string server_resp;
  {
    std::lock_guard<std::mutex> lk(g_sock_mtx);
    const bool ok_rpc = request_response_json_line(INFER_SOCK_PATH, req.str(), &server_resp);
    if (!ok_rpc) {
      std::cerr << "bnnpref: RPC failed (socket " << INFER_SOCK_PATH << ")\n";
      return metadata_in;
    }
  }

  // Extract "0x..." tokens from JSON and issue prefetches
  try {
    static const std::regex hex_rx("0x[0-9a-fA-F]+");
    auto it = std::sregex_iterator(server_resp.begin(), server_resp.end(), hex_rx);
    auto end = std::sregex_iterator();
    for (; it != end; ++it) {
      const std::string token = (*it).str();
      unsigned long long raw = std::stoull(token, nullptr, 0);
      prefetch_line(champsim::address{raw}, /*fill_this_level=*/true, metadata_in);
    }
  } catch (...) {
    // ignore parse errors
  }

  return metadata_in;
}

uint32_t bnnpref::prefetcher_cache_fill(champsim::address /*addr*/, long /*set*/, long /*way*/,
                                        uint8_t /*prefetch*/, champsim::address /*evicted_addr*/,
                                        uint32_t metadata_in)
{
  return metadata_in;
}

