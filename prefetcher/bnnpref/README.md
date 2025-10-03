# BNN-Prefetcher for ChampSim

A lightweight ML-driven prefetcher for ChampSim that predicts next cache-line addresses using a compact Bi-Mixer model served from a small Python inference server over a UNIX socket. This repo also includes a convenience runner, `run_options.py`, to sweep classic prefetchers across L1D/L2C for side-by-side comparisons (and to parse results automatically).

## Contents

```
.
├── run_options.py                 # Batch runner for L1D/L2C prefetcher sweeps
├── parse_runs.py                  # Summarizes simulator outputs into a CSV (called by run_options.py)
├── champsim_config.json           # Simulator config edited by run_options.py
├── run_champsim.sh                # Wrapper that runs ChampSim with the current config
├── sim_output/                    # (Created) All run logs/results land here
└── prefetcher/
    └── bnnpref/
        ├── bnnpref.h, bnnpref.cc  # C++ client in ChampSim
        └── model/
            ├── infer.py           # One-shot inference helper
            ├── infer_server.py    # Persistent inference server (UNIX socket)
            └── files/weights/
                └── *.pt           # Trained checkpoint(s) with vocab & k-means
```

---

## How it works (high level)

* **Client (C++ in ChampSim):** keeps a sliding history of memory events and, after warmup, sends a compact JSON with the last `HISTORY_LENGTH` events via a UNIX socket (`/tmp/bnnpref.sock` by default).
* **Server (Python):** loads a checkpoint (includes k-means centers and delta vocab), maps history to cluster/local-delta ids, runs the Bi-Mixer, decodes top-K delta logits to absolute addresses, and returns them as hex strings.
* **Prefetching:** the client parses returned `0x...` addresses and issues `prefetch_line(...)` calls.

> Note: The client intentionally **skips early requests** (warmup) to avoid noisy behavior. See `SKIP_FIRST_N` in `bnnpref.cc` if you want to change it.

---

## Requirements

* **ChampSim** built for your platform (`make -j`).
* **Python 3.8+** with:

  * `torch` (CPU OK; GPU optional), `numpy`
* (Optional) CUDA-enabled PyTorch if running the server on a GPU.

Quick Python setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy
```

---

## Quick Start

### 1) Make sure ChampSim is built (Follow instructions in `README.md`).

### 2) Run the server (in a separate terminal):

```bash
# From the repo root
./start_server.sh
```

### 3) Run the batch runner:

```bash
# From the repo root
python3 run_options.py --prefetchers bnnpref --l2c_only
```

Once that finishes, you can open `reports/` to see the results.

---

## `run_options.py` details

### What it does

* Verifies/Downloads the default trace file if missing.
* Edits `champsim_config.json` to set the chosen L1D/L2C prefetchers.
* Calls `./run_champsim.sh <out_path>` for each combination.
* Times each simulation and prints duration.
* Calls `parse_runs()` to create a consolidated summary (CSV/TSV) of the runs.

### CLI

```
usage: run_options.py [-h] [--config CONFIG] [--output OUTPUT]
                      [--all_prefetchers] [--l1d_only] [--l2c_only]
                      [--prefetchers PREFETCHERS [PREFETCHERS ...]]
```

* `--config` (default: `champsim_config.json`)
  Path to the JSON config that contains `"L1D": {"prefetcher": ...}, "L2C": {"prefetcher": ...}`.

* `--output` (default: `sim_output`)
  Directory to store run outputs.

* `--all_prefetchers`
  (Flag) Run both sweeps using the given `--prefetchers` list.

* `--l1d_only` / `--l2c_only`
  Mutually exclusive. Run a sweep with L1D (L2C set to `"no"`) or L2C (L1D set to `"no"`).

* `--prefetchers ...`
  Override the default list. Example:

  ```
  --prefetchers no next_line ip_stride
  ```

> If `--prefetchers` is provided without `--all_prefetchers`, it applies to the selected sweep.

### Output layout

Each simulation is invoked as:

```
./run_champsim.sh sim_output/L1D_<L1D_NAME>_L2C_<L2C_NAME>.txt
```

You’ll get one file per run, plus whatever standard ChampSim logs/stats `run_champsim.sh` emits (e.g., per-component stats).

### Parsing results

`run_options.py` finishes by running:

```python
from parse_runs import parse_runs
parse_runs()
```

Ensure `parse_runs.py`:

* Scans `sim_output/` (or your `--output` dir).
* Produces a summary file (e.g., `sim_output/summary.csv`) with key metrics like IPC, MPKI, MISS/HIT, MSHR merges, prefetch accuracy/usefulness, etc.

---

## Running ChampSim manually with `bnnpref`

Once your server is running:

```bash
# Ensure champsim_config.json (or equivalent) has:
#   "L1D": {"prefetcher": "bnnpref"}  or  "L2C": {"prefetcher": "bnnpref"}

./run_champsim.sh sim_output/my_bnnpref_run.txt
```

If the server is unreachable, you will see:

```
RPC failed (socket /tmp/bnnpref.sock)
```

The simulation continues; the prefetcher simply won’t issue ML-based prefetches.

---

## Debugging & Offline Inference

**One-shot inference** for a saved history CSV:

```bash
python3 prefetcher/bnnpref/model/infer.py \
  --input_file /path/to/history.csv \
  --output_file /tmp/preds.txt \
  --ckpt prefetcher/bnnpref/model/files/weights/prefetch_mixer_ckpt_best.pt
```

**Manual socket test** (newline-delimited JSON):

```bash
nc -U /tmp/bnnpref.sock
# then paste one line:
{"hist_inline":{"addr":["0x..."],"ip":["0x..."],"hit":[0,1,0,...]},"topk":10}
# server replies with a single JSON line: {"pred": ["0x....", ...]}
```

---

## Tuning knobs

* **Client (C++)** in `bnnpref.h/.cc`:

  * `HISTORY_LENGTH` (default 64)
  * `PREF_TOPK` (e.g., 10)
  * `BLOCK_BYTES` (64)
  * `INFER_SOCK_PATH` (`/tmp/bnnpref.sock`)
  * `SKIP_FIRST_N` (warmup requests to skip)

* **Server (Python)** flags in `infer_server.py`:

  * `--ckpt PATH` (required)
  * `--device {cpu,cuda}` (optional; auto-detects if omitted)
  * `--socket /tmp/bnnpref.sock` (optional)
  * Request-level overrides: `hist`, `topk`, `block_bytes` (if your client sends them)

---

## Troubleshooting

* **Trace missing / download fails**
  `run_options.py` will attempt to download `benchbase-tpcc.champsim.trace.gz` to `traces/`. If your machine blocks `wget`, download manually and place it at:

  ```
  traces/benchbase-tpcc.champsim.trace.gz
  ```

* **Server socket errors / hangs**

  * Confirm the server prints “listening on /tmp/bnnpref.sock”.
  * Remove stale sockets: `rm -f /tmp/bnnpref.sock`.
  * Ensure versions of PyTorch/NumPy are consistent with your checkpoint.

* **Checkpoint load error**

  * Use the recommended PyTorch 2.x on CPU first.
  * Verify the checkpoint path is correct and readable.

* **No prefetches issued**

  * You may still be in warmup (`SKIP_FIRST_N`).
  * The server may be down or returning empty predictions.

---

## FAQ

**Q: Can I run sweeps that include `bnnpref`?**
A: Yes—add `"bnnpref"` to `--prefetchers` and ensure the server is running.

**Q: Where is the summary CSV?**
A: `parse_runs.py` should emit something like `sim_output/summary.csv`. If your version writes a different path/name, adjust the script or check its console output.

**Q: How do I change L1D/L2C defaults?**
A: Edit `champsim_config.json`. `run_options.py` updates only the `"prefetcher"` fields.

---

## Example Workflows

**Compare classic prefetchers on L1D, keep L2C off:**

```bash
python3 run_options.py --l1d_only --prefetchers no next_line ip_stride spp_dev va_ampm_lite
```

**Single bnnpref run (manual):**

```bash
# Terminal 1 (server)
python3 prefetcher/bnnpref/model/infer_server.py --ckpt prefetcher/bnnpref/model/files/weights/prefetch_mixer_ckpt_best.pt

# Terminal 2 (sim)
# Set L1D or L2C prefetcher to "bnnpref" in champsim_config.json, then:
./run_champsim.sh sim_output/bnnpref_L1D.txt
```

---

## Citation / Credits

The BNN prefetcher integration and runner tooling were added in this fork. The checkpoints and training pipeline are maintained separately. If you use this in academic work, please cite ChampSim and your own prefetcher methodology appropriately.

For questions or training scripts, open an issue or PR.
