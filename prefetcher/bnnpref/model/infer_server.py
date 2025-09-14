#!/usr/bin/env python3
# BiMixer persistent inference server (UNIX socket, newline-framed JSON)
# - Global server-side progress bar: progress = requests_made / MAX_REQ
# - Robust against client disconnects
# - Backward compatible response: single JSON line per request

import os, sys, socket, json, traceback, argparse
from typing import Dict, Optional
import numpy as np
import torch

# Import helpers from infer.py (sibling file)
from infer import (
    PrefetchBiMixer,
    parse_history_csv,
    assign_clusters_1d,
    build_delta_in_ids,
    logits_to_topk_deltas,
    atomic_write_lines,
)

# ────────────────────────── Globals for progress ──────────────────────────
REQ_COUNT: int = 0
REQ_MAX: int = 100
PROGRESS_ENABLED: bool = True
PROGRESS_LABEL: str = "bnnpref"
BAR_WIDTH: int = 40
_LAST_DRAWN_PCT: int = -1
_DONE_PRINTED: bool = False
_LAST_DRAWN_COUNT: int = -1   # ← NEW: track last count, not just percent
_IS_TTY: bool = sys.stdout.isatty()  # ← NEW: detect TTY vs file

def draw_global_progress():
    """Update global progress. Redraw if count OR percent changed.
       In non-TTY (logs/nohup), print one line per request."""
    global _LAST_DRAWN_PCT, _LAST_DRAWN_COUNT, _DONE_PRINTED
    if not PROGRESS_ENABLED or REQ_MAX <= 0:
        return

    count = min(REQ_COUNT, REQ_MAX)
    pct = int(100 * count / REQ_MAX)
    pct = max(0, min(100, pct))

    # Only redraw if something actually changed
    changed = (count != _LAST_DRAWN_COUNT) or (pct != _LAST_DRAWN_PCT) or (pct == 100 and not _DONE_PRINTED)
    if not changed:
        return

    _LAST_DRAWN_COUNT = count
    _LAST_DRAWN_PCT = pct

    if _IS_TTY:
        filled = int(BAR_WIDTH * pct / 100.0)
        bar = "#" * filled + "-" * (BAR_WIDTH - filled)
        sys.stdout.write(f"\r[{PROGRESS_LABEL}] [{bar}] {pct:3d}%  ({count}/{REQ_MAX})")
        sys.stdout.flush()
        if pct >= 100 and not _DONE_PRINTED:
            sys.stdout.write("\n")
            sys.stdout.flush()
            _DONE_PRINTED = True
    else:
        # Non-interactive: write a new line per request
        sys.stdout.write(f"[{PROGRESS_LABEL}] {pct:3d}%  ({count}/{REQ_MAX})\n")
        sys.stdout.flush()
        if pct >= 100 and not _DONE_PRINTED:
            _DONE_PRINTED = True


# ────────────────────────── Engine build ──────────────────────────
def build_engine(ckpt_path: str, history_len: int, device_str: str):
    device = torch.device("cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    eng: Dict[str, object] = {}
    eng["CLUSTERS"]    = int(cfg["CLUSTERS"])
    eng["TOP_PER_CL"]  = int(cfg["TOP_PER_CL"])
    eng["N_DELTA_IN"]  = int(cfg["n_delta_inputs_per_cluster"])
    eng["N_CLASSES"]   = int(cfg["n_classes_per_cluster"])
    PC_EMB             = int(cfg["pc_emb"])
    DELTA_EMB          = int(cfg["delta_emb"])
    CL_EMB             = int(cfg["cluster_emb"])
    MIXER_DIM          = int(cfg["mixer_dim"])
    MIXER_DEPTH        = int(cfg["mixer_depth"])
    TOK_DIM            = int(cfg["token_mlp_dim"])
    CHN_DIM            = int(cfg["channel_mlp_dim"])
    SEMI_BIN           = bool(cfg["semi_binary"])
    DROPOUT            = float(cfg["dropout"])
    eng["USE_TAIL"]    = bool(cfg["use_tail_buckets"])
    eng["MAG_BINS"]    = int(cfg["tail_mag_bins"]) if eng["USE_TAIL"] else 0
    eng["history_len"] = int(history_len)
    eng["device"]      = device

    eng["pc2id"] = {int(k): int(v) for k, v in ckpt["pc2id"].items()}
    eng["id2delta_per"] = {int(c): {int(i): int(d) for i, d in m.items()} for c, m in ckpt["id2delta_per"].items()}
    eng["delta_out2id_per"] = {int(c): {int(d): int(i) for d, i in m.items()} for c, m in ckpt["delta_out2id_per"].items()}
    eng["bucket_fallbacks"] = {int(c): {int(b): [int(x) for x in arr] for b, arr in m.items()} for c, m in ckpt["bucket_fallbacks"].items()}
    eng["centers_flat"] = np.asarray(ckpt["kmeans_centers"], dtype=np.float64).reshape(-1)

    n_pcs_with_unk = int(cfg["n_pcs_with_unk"])
    model = PrefetchBiMixer(
        n_clusters=eng["CLUSTERS"],
        n_pcs_with_unk=n_pcs_with_unk,
        n_delta_inputs_per_cluster=eng["N_DELTA_IN"],
        pc_emb=PC_EMB, delta_emb=DELTA_EMB, cluster_emb=CL_EMB,
        dim=MIXER_DIM, mixer_depth=MIXER_DEPTH,
        token_mlp_dim=TOK_DIM, channel_mlp_dim=CHN_DIM,
        dropout=DROPOUT, n_classes_per_cluster=eng["N_CLASSES"],
        semi_binary=SEMI_BIN, seq_len=eng["history_len"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    eng["model"] = model
    return eng


# ────────────────────────── Helpers ──────────────────────────
def recv_lines(sock: socket.socket, buf: bytearray):
    """Yield complete newline-terminated lines from a socket."""
    while True:
        nl = buf.find(b'\n')
        if nl != -1:
            line = bytes(buf[:nl])
            del buf[:nl+1]
            yield line
            continue
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk

def to_int_list(xs):
    """Accept '0x..'/'0X..' or decimal strings or ints, return Python-int list."""
    out = []
    for i, x in enumerate(xs):
        try:
            if isinstance(x, str):
                out.append(int(x, 0))  # base=0 handles "0x.." and decimal
            else:
                out.append(int(x))
        except Exception as e:
            print(f"[error] to_int_list: bad element at index {i}: {x!r}", file=sys.stderr)
            raise ValueError(f"to_int_list: bad element at index {i}: {x!r}") from e
    return out

def safe_send(conn: socket.socket, payload_str: str) -> bool:
    """Send one JSON line safely; swallow disconnects and signal caller to stop."""
    try:
        conn.sendall((payload_str + "\n").encode("utf-8"))
        return True
    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        print(f"[warn] client disconnected before send: {e}")
        return False


# ────────────────────────── Inference (inline / csv) ──────────────────────────
def infer_once_inline(eng, hist_dict, out_path: Optional[str], topk: int = 10, block_bytes: int = 64):
    # parse
    blocks_all = np.array(to_int_list(hist_dict.get("addr", [])), dtype=object)
    pcs_all    = np.array(to_int_list(hist_dict.get("ip",   [])), dtype=object)
    hits_all   = np.array([int(h) for h in hist_dict.get("hit", [])], dtype=np.uint8)

    # miss filter
    miss_mask = (hits_all == 0)
    if miss_mask.sum() == 0:
        if out_path:
            atomic_write_lines(out_path, [])
        return {"ok": True, "n": 0, "pred": []}

    blocks = blocks_all[miss_mask]
    pcs    = pcs_all[miss_mask]

    # align to block size if power-of-two
    if block_bytes and (block_bytes & (block_bytes - 1)) == 0:
        mask = ~((block_bytes) - 1) & ((1 << 64) - 1)
        blocks = np.array([int(b) & mask for b in blocks.tolist()], dtype=object)

    # cluster
    cls = assign_clusters_1d(blocks, eng["centers_flat"])
    if cls.size == 0:
        if out_path:
            atomic_write_lines(out_path, [])
        return {"ok": True, "n": 0, "pred": []}

    cl_curr = int(cls[-1])

    # slice history for current cluster
    idx_c = np.nonzero(cls == cl_curr)[0]
    if len(idx_c) == 0:
        if out_path:
            atomic_write_lines(out_path, [])
        return {"ok": True, "n": 0, "cluster": cl_curr, "pred": []}
    take_idx = idx_c[-eng["history_len"]:]
    seq_blocks = [int(blocks[j]) for j in take_idx]
    seq_pcs    = [int(pcs[j])    for j in take_idx]
    if len(seq_blocks) < eng["history_len"]:
        need = eng["history_len"] - len(seq_blocks)
        pad_block = seq_blocks[0]
        seq_blocks = [pad_block]*need + seq_blocks
        seq_pcs    = [0]*need + seq_pcs

    # map ids
    local_deltas = [0] + [seq_blocks[i] - seq_blocks[i-1] for i in range(1, len(seq_blocks))]
    pc_ids = np.array([eng["pc2id"].get(int(x), 0) for x in seq_pcs], dtype=np.int64)
    delta_in_ids = build_delta_in_ids(local_deltas, cl_curr, eng["delta_out2id_per"],
                                      eng["TOP_PER_CL"], eng["USE_TAIL"], eng["MAG_BINS"])
    dglob = (cl_curr * eng["N_DELTA_IN"]) + np.asarray(delta_in_ids, dtype=np.int64)

    # tensors
    device = eng["device"]
    cl_t = torch.tensor([cl_curr], dtype=torch.long, device=device)
    pc_t = torch.tensor(pc_ids[None, :], dtype=torch.long, device=device)
    dg_t = torch.tensor(dglob[None, :], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = eng["model"](cl_t, pc_t, dg_t)

    # decode
    cand_deltas = logits_to_topk_deltas(logits, cl_curr, eng["id2delta_per"],
                                        eng["bucket_fallbacks"], topk, eng["TOP_PER_CL"])
    last_block = seq_blocks[-1]
    pred_blocks, seen = [], set()
    for d in cand_deltas:
        b = last_block + int(d)
        if b < 0:
            continue
        if b not in seen:
            pred_blocks.append(b); seen.add(b)

    hex_lines = [f"0x{b:x}" for b in pred_blocks]
    if out_path:
        atomic_write_lines(out_path, hex_lines)
    return {"ok": True, "n": len(hex_lines), "cluster": cl_curr, "pred": hex_lines}

def infer_once(eng, in_csv: str, out_path: str, topk: int = 10, block_bytes: int = 64):
    # CSV fallback path (kept for compatibility)
    blocks_all, pcs_all, hits_all = parse_history_csv(in_csv)
    miss_mask = (hits_all == 0)
    if miss_mask.sum() == 0:
        atomic_write_lines(out_path, [])
        return {"ok": True, "n": 0}

    blocks = blocks_all[miss_mask]
    pcs    = pcs_all[miss_mask]
    if block_bytes and (block_bytes & (block_bytes - 1)) == 0:
        mask = ~((block_bytes) - 1) & ((1 << 64) - 1)
        blocks = np.array([int(b) & mask for b in blocks.tolist()], dtype=object)

    cls = assign_clusters_1d(blocks, eng["centers_flat"])
    if cls.size == 0:
        atomic_write_lines(out_path, [])
        return {"ok": True, "n": 0}

    cl_curr = int(cls[-1])
    idx_c = np.nonzero(cls == cl_curr)[0]
    if len(idx_c) == 0:
        atomic_write_lines(out_path, [])
        return {"ok": True, "n": 0, "cluster": cl_curr}

    take_idx = idx_c[-eng["history_len"]:]
    seq_blocks = [int(blocks[j]) for j in take_idx]
    seq_pcs    = [int(pcs[j])    for j in take_idx]
    if len(seq_blocks) < eng["history_len"]:
        need = eng["history_len"] - len(seq_blocks)
        pad_block = seq_blocks[0]
        seq_blocks = [pad_block]*need + seq_blocks
        seq_pcs    = [0]*need + seq_pcs

    local_deltas = [0] + [seq_blocks[i] - seq_blocks[i-1] for i in range(1, len(seq_blocks))]
    pc_ids = np.array([eng["pc2id"].get(int(x), 0) for x in seq_pcs], dtype=np.int64)
    delta_in_ids = build_delta_in_ids(local_deltas, cl_curr, eng["delta_out2id_per"],
                                      eng["TOP_PER_CL"], eng["USE_TAIL"], eng["MAG_BINS"])
    dglob = (cl_curr * eng["N_DELTA_IN"]) + np.asarray(delta_in_ids, dtype=np.int64)

    device = eng["device"]
    cl_t = torch.tensor([cl_curr], dtype=torch.long, device=device)
    pc_t = torch.tensor(pc_ids[None, :], dtype=torch.long, device=device)
    dg_t = torch.tensor(dglob[None, :], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = eng["model"](cl_t, pc_t, dg_t)  # (1, C)

    cand_deltas = logits_to_topk_deltas(logits, cl_curr, eng["id2delta_per"],
                                        eng["bucket_fallbacks"], topk, eng["TOP_PER_CL"])
    last_block = seq_blocks[-1]
    pred_blocks, seen = [], set()
    for d in cand_deltas:
        b_signed = last_block + int(d)
        if b_signed < 0:
            continue
        if b_signed not in seen:
            pred_blocks.append(b_signed); seen.add(b_signed)

    hex_lines = [f"0x{b:x}" for b in pred_blocks]
    atomic_write_lines(out_path, hex_lines)
    return {"ok": True, "n": len(hex_lines), "cluster": cl_curr}


# ────────────────────────── Persistent server ──────────────────────────
def serve(sock_path: str, ckpt_path: str, history_len: int, device: str, threads: int,
          max_req: int, progress_label: str, no_progress: bool):
    # declare all globals up front (single place)
    global REQ_MAX, PROGRESS_LABEL, PROGRESS_ENABLED, _LAST_DRAWN_PCT, _DONE_PRINTED, REQ_COUNT

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)

    REQ_MAX = max(1, int(max_req))
    PROGRESS_LABEL = progress_label
    PROGRESS_ENABLED = not no_progress
    _LAST_DRAWN_PCT = -1
    _DONE_PRINTED = False
    if PROGRESS_ENABLED:
        # draw initial empty bar
        draw_global_progress()

    eng = build_engine(ckpt_path, history_len, device)

    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(64)
    os.chmod(sock_path, 0o666)

    print(f"\n[infer_server] listening on {sock_path}")
    print(f"[infer_server] device={eng['device']}, history_len={eng['history_len']}, max_req={REQ_MAX}")

    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 16)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 16)
        try:
            buf = bytearray()
            for raw in recv_lines(conn, buf):
                # Parse one JSON request per line
                try:
                    req = json.loads(raw.decode("utf-8"))
                except Exception as e:
                    if not safe_send(conn, json.dumps({"ok": False, "err": f"bad_json: {e}"})):
                        break
                    continue

                # Optional per-request override for max requests
                try:
                    if "max_req" in req:
                        mx = int(req["max_req"])
                        if mx > 0:
                            REQ_MAX = mx
                            _LAST_DRAWN_PCT = -1
                            _DONE_PRINTED = False
                            draw_global_progress()
                except Exception:
                    pass

                if req.get("cmd") == "PING":
                    if not safe_send(conn, json.dumps({"ok": True, "pong": True})):
                        break
                    continue

                in_csv   = req.get("in")
                out_txt  = req.get("out")         # optional for back-compat
                hist_obj = req.get("hist_inline") # fast path
                topk     = int(req.get("topk", 10))
                hist     = int(req.get("hist", eng["history_len"]))
                blk      = int(req.get("block_bytes", 64))

                # Count this request (non-PING) and update global progress
                REQ_COUNT += 1
                draw_global_progress()

                # per-request history override
                eng["history_len"] = hist

                try:
                    if hist_obj is not None:
                        resp = infer_once_inline(eng, hist_obj, out_txt, topk=topk, block_bytes=blk)
                    elif in_csv and out_txt:
                        resp = infer_once(eng, in_csv, out_txt, topk=topk, block_bytes=blk)
                    else:
                        resp = {"ok": False, "err": "missing ('hist_inline') OR ('in' and 'out')"}
                except Exception as e:
                    resp = {"ok": False, "err": str(e), "trace": traceback.format_exc()}

                # Final result line (always send one terminal line)
                if not safe_send(conn, json.dumps(resp)):
                    break
        finally:
            conn.close()


# ────────────────────────── CLI ──────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BiMixer persistent inference server")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt_dir = os.path.join(script_dir, "files", "weights")
    ckpt_default = None
    if os.path.isdir(default_ckpt_dir):
        pts = [os.path.join(default_ckpt_dir, f) for f in os.listdir(default_ckpt_dir) if f.endswith(".pt")]
        if pts: ckpt_default = sorted(pts)[0]

    # Default socket chosen to match your bnnpref usage
    parser.add_argument("--sock", default="/tmp/bnnpref.sock", help="UNIX socket path")
    parser.add_argument("--ckpt", default=ckpt_default, required=(ckpt_default is None), help="Path to .pt checkpoint")
    parser.add_argument("--history_len", type=int, default=64)
    parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=1, help="intra-op threads for torch/BLAS")

    # Global progress controls
    parser.add_argument("--max-req", type=int, default=500000000, help="Total number of requests after which progress = 100%")
    parser.add_argument("--progress-label", type=str, default="bnnpref", help="Label printed next to the progress bar")
    parser.add_argument("--no-progress", action="store_true", help="Disable server-side progress bar")

    args = parser.parse_args()

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    serve(args.sock, args.ckpt, args.history_len, dev, args.threads,
          max_req=args.max_req, progress_label=args.progress_label, no_progress=args.no_progress)


if __name__ == "__main__":
    main()

