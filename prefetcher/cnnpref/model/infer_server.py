#!/usr/bin/env python3
# BiMixer persistent inference server (UNIX socket)

import os, sys, socket, json, time, traceback, argparse
import numpy as np
import torch

# Import helpers from your infer.py (sibling file)
from infer import (
    PrefetchBiMixer,
    parse_history_csv,
    assign_clusters_1d,
    build_delta_in_ids,
    logits_to_topk_deltas,
    atomic_write_lines,
)

def build_engine(ckpt_path: str, history_len: int, device_str: str):
    device = torch.device("cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    eng = {}
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

def to_int_list(xs):
    # convert ["0x..", ...] or [123, ...] → [int, ...]
    out = []
    for i, x in enumerate(xs):
        try:
            out.append(int(x, 16) if isinstance(x, str) else int(x))
        except Exception as e:
            print(f"[error] to_int_list: bad element at index {i}: {x!r}", file=sys.stderr)
            raise ValueError(f"to_int_list: bad element at index {i}: {x!r}") from e
    return out

def infer_once_inline(eng, hist_dict, out_path: str, topk: int = 10, block_bytes: int = 64):
    # Expect: hist_dict = {"addr":[...], "ip":[...], "hit":[...]} (strings "0x..." or ints)
    blocks_all = np.array(to_int_list(hist_dict.get("addr", [])), dtype=object)
    pcs_all    = np.array(to_int_list(hist_dict.get("ip",   [])), dtype=object)
    hits_all   = np.array([int(h) for h in hist_dict.get("hit", [])], dtype=np.int8)

    # Reuse the same body as infer_once from here down:
    miss_mask = (hits_all == 0)
    if miss_mask.sum() == 0:
        atomic_write_lines(out_path, [])
        print(f"[warn] no misses in history")   
        return {"ok": True, "n": 0}

    blocks = blocks_all[miss_mask]
    pcs    = pcs_all[miss_mask]

    if block_bytes and (block_bytes & (block_bytes - 1)) == 0:
        mask = ~((block_bytes) - 1) & ((1 << 64) - 1)
        blocks = np.array([int(b) & mask for b in blocks.tolist()], dtype=object)

    cls = assign_clusters_1d(blocks, eng["centers_flat"])
    if cls.size == 0:
        atomic_write_lines(out_path, [])
        print(f"[warn] no valid history after filtering")
        return {"ok": True, "n": 0}

    cl_curr = int(cls[-1])
    idx_c = np.nonzero(cls == cl_curr)[0]
    if len(idx_c) == 0:
        atomic_write_lines(out_path, [])
        print(f"[warn] no history in current cluster {cl_curr}")
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
        logits = eng["model"](cl_t, pc_t, dg_t)

    cand_deltas = logits_to_topk_deltas(logits, cl_curr, eng["id2delta_per"],
                                        eng["bucket_fallbacks"], topk, eng["TOP_PER_CL"])
    last_block = seq_blocks[-1]
    pred_blocks, seen = [], set()
    for d in cand_deltas:
        b = last_block + int(d)
        if b < 0:  # drop underflow
            continue
        if b not in seen:
            pred_blocks.append(b); seen.add(b)

    hex_lines = [f"0x{b:x}" for b in pred_blocks]
    atomic_write_lines(out_path, hex_lines)
    return {"ok": True, "n": len(hex_lines), "cluster": cl_curr}


def infer_once(eng, in_csv: str, out_path: str, topk: int = 10, block_bytes: int = 64):
    t0 = time.time()
    print(f"[infer_once] in='{in_csv}' out='{out_path}' topk={topk} block_bytes={block_bytes} ...", end=' ', flush=True)

    blocks_all, pcs_all, hits_all = parse_history_csv(in_csv)
    miss_mask = (hits_all == 0)
    if miss_mask.sum() == 0:
        print(f"[warn] no misses in history")
        atomic_write_lines(out_path, [])
        return {"ok": True, "n": 0, "ms": (time.time()-t0)*1e3}

    blocks = blocks_all[miss_mask]
    pcs    = pcs_all[miss_mask]

    if block_bytes and (block_bytes & (block_bytes - 1)) == 0:
        mask = ~((block_bytes) - 1) & ((1 << 64) - 1)
        blocks = np.array([int(b) & mask for b in blocks.tolist()], dtype=object)

    cls = assign_clusters_1d(blocks, eng["centers_flat"])
    if cls.size == 0:
        atomic_write_lines(out_path, [])
        print(f"[warn] no valid history after filtering")
        return {"ok": True, "n": 0, "ms": (time.time()-t0)*1e3}

    cl_curr = int(cls[-1])

    idx_c = np.nonzero(cls == cl_curr)[0]
    if len(idx_c) == 0:
        atomic_write_lines(out_path, [])
        print(f"[warn] no history in current cluster {cl_curr}")
        return {"ok": True, "n": 0, "cluster": cl_curr, "ms": (time.time()-t0)*1e3}

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

    # ↓↓↓ FIX: use eng['device'], not model.device
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
    print(f"Written to {out_path}: {len(hex_lines)} lines, cluster {cl_curr}, in {(time.time()-t0)*1e3:.1f} ms")

    return {"ok": True, "n": len(hex_lines), "cluster": cl_curr, "ms": (time.time()-t0)*1e3}

def serve(sock_path: str, ckpt_path: str, history_len: int, device: str, threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)

    eng = build_engine(ckpt_path, history_len, device)

    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(32)
    os.chmod(sock_path, 0o666)

    print(f"[infer_server] listening on {sock_path}")
    print(f"[infer_server] device={eng['device']}, history_len={eng['history_len']}")

    while True:
        conn, _ = srv.accept()
        try:
            # read full request
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            # parse JSON
            try:
                req = json.loads(data.decode("utf-8")) if data else {}
            except Exception as e:
                conn.sendall(json.dumps({"ok": False, "err": f"bad_json: {e}"}).encode("utf-8"))
                continue

            # ping
            if req.get("cmd") == "PING":
                conn.sendall(json.dumps({"ok": True, "pong": True}).encode("utf-8"))
                continue

            # params
            in_csv   = req.get("in")
            out_txt  = req.get("out")
            hist_obj = req.get("hist_inline")  # NEW: inline fast path
            topk     = int(req.get("topk", 10))
            hist     = int(req.get("hist", eng["history_len"]))
            blk      = int(req.get("block_bytes", 64))

            # allow per-request history length override (we pad/truncate inside)
            if hist != eng["history_len"]:
                eng["history_len"] = hist

            # dispatch
            if hist_obj is not None and out_txt:
                try:
                    resp = infer_once_inline(eng, hist_obj, out_txt, topk=topk, block_bytes=blk)
                except Exception as e:
                    resp = {"ok": False, "err": str(e), "trace": traceback.format_exc()}
            elif in_csv and out_txt:
                try:
                    resp = infer_once(eng, in_csv, out_txt, topk=topk, block_bytes=blk)
                except Exception as e:
                    resp = {"ok": False, "err": str(e), "trace": traceback.format_exc()}
            else:
                resp = {"ok": False, "err": "missing ('hist_inline' and 'out') OR ('in' and 'out')"}

            conn.sendall(json.dumps(resp).encode("utf-8"))
        finally:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="BiMixer persistent inference server")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt_dir = os.path.join(script_dir, "files", "weights")
    ckpt_default = None
    if os.path.isdir(default_ckpt_dir):
        pts = [os.path.join(default_ckpt_dir, f) for f in os.listdir(default_ckpt_dir) if f.endswith(".pt")]
        if pts: ckpt_default = sorted(pts)[0]

    parser.add_argument("--sock", default="/tmp/cnnpref.sock", help="UNIX socket path")
    parser.add_argument("--ckpt", default=ckpt_default, required=(ckpt_default is None), help="Path to .pt checkpoint")
    parser.add_argument("--history_len", type=int, default=64)
    parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=1, help="intra-op threads for torch/BLAS")
    args = parser.parse_args()

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    serve(args.sock, args.ckpt, args.history_len, dev, args.threads)

if __name__ == "__main__":
    main()

