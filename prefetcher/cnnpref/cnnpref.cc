#include "cnnpref.h"
#include <iostream>
#include <string>

uint32_t cnnpref::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                      uint32_t metadata_in)
{
  int64_t block = static_cast<int64_t>((addr >> champsim::data::bits{6}).value());
  int64_t delta = (last_block != -1) ? block - last_block : 0;
  last_block = block;

  history.emplace_back(static_cast<uint64_t>(ip), delta);
  if (history.size() < HISTORY_LENGTH)
    return metadata_in;
  if (history.size() > HISTORY_LENGTH)
    history.pop_front();

  // Format the input: "pc1,pc2,...;d1,d2,..."
  std::stringstream ss_pc, ss_delta;
  for (const auto& [pc, d] : history) {
    ss_pc << pc << ",";
    ss_delta << d << ",";
  }
  std::string pcs_str = ss_pc.str();
  std::string deltas_str = ss_delta.str();
  pcs_str.pop_back();      // remove last comma
  deltas_str.pop_back();

  std::stringstream cmd;
  cmd << "python3 infr_single.py <<< \"" << pcs_str << ";" << deltas_str << "\"";

  FILE* pipe = popen(cmd.str().c_str(), "r");
  if (!pipe) {
    std::cerr << "Failed to run inference script\n";
    return metadata_in;
  }

  char buffer[64];
  std::string output;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    output += buffer;
  pclose(pipe);

  try {
    int64_t pred_delta = std::stoll(output);
    champsim::block_number pf_block = static_cast<champsim::block_number>(block + pred_delta);
    prefetch_line(champsim::address{pf_block}, true, metadata_in);
  } catch (...) {
    std::cerr << "Error parsing prediction output: " << output << std::endl;
  }

  return metadata_in;
}

