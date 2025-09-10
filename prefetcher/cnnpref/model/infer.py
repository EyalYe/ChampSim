#!/usr/bin/env python3
import os
import math
import argparse
import tempfile
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────────────── Model building blocks ──────────────────────────
class BinaryLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        W = self.W
        alpha = W.abs().mean(dim=1, keepdim=True)  # per-output scale
        Wb = alpha * torch.sign(W)
        Wb[torch.isnan(Wb)] = 0.0
        W_ste = W + (Wb - W).detach()
        return F.linear(x, W_ste, self.bias)

class PreSign(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ln(x)
        y = x.sign()
        y[y == 0] = 1
        return x + (y - x).detach()

class MixerBlock(nn.Module):
    def __init__(self, seq_len, dim, token_mlp_dim, channel_mlp_dim, dropout=0.0, semi_binary=True):
        super().__init__()
        self.pre_tok = PreSign(dim)
        self.tok_fc1 = BinaryLinear(seq_len, token_mlp_dim, bias=False)
        self.tok_fc2 = (nn.Linear(token_mlp_dim, seq_len, bias=False)
                        if semi_binary else BinaryLinear(token_mlp_dim, seq_len, bias=False))
        self.pre_chn = PreSign(dim)
        self.chn_fc1 = BinaryLinear(dim, channel_mlp_dim, bias=False)
        self.chn_fc2 = (nn.Linear(channel_mlp_dim, dim, bias=False)
                        if semi_binary else BinaryLinear(channel_mlp_dim, dim, bias=False))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, L, D)
        y = self.pre_tok(x)
        y = y.transpose(1, 2)
        y = self.tok_fc2(F.gelu(self.tok_fc1(y)))
        y = y.transpose(1, 2)
        x = x + self.drop(y)

        y = self.pre_chn(x)
        y = self.chn_fc2(F.gelu(self.chn_fc1(y)))
        x = x + self.drop(y)
        return x

class PrefetchBiMixer(nn.Module):
    def __init__(self, n_clusters, n_pcs_with_unk, n_delta_inputs_per_cluster,
                 pc_emb, delta_emb, cluster_emb,
                 dim, mixer_depth, token_mlp_dim, channel_mlp_dim,
                 dropout, n_classes_per_cluster, semi_binary, seq_len):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_classes  = n_classes_per_cluster

        self.cluster_embed = nn.Embedding(n_clusters, cluster_emb)
        self.pc_embed      = nn.Embedding(n_pcs_with_unk, pc_emb)
        self.delta_embed   = nn.Embedding(n_clusters * n_delta_inputs_per_cluster, delta_emb)

        in_dim = pc_emb + delta_emb
        self.in_proj = nn.Linear(in_dim, dim, bias=False)

        self.blocks = nn.ModuleList([
            MixerBlock(seq_len=seq_len, dim=dim,
                       token_mlp_dim=token_mlp_dim,
                       channel_mlp_dim=channel_mlp_dim,
                       dropout=dropout,
                       semi_binary=semi_binary)
            for _ in range(mixer_depth)
        ])

        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(dropout)

        self.heads = nn.ModuleList([
            nn.Linear(2*dim + cluster_emb, n_classes_per_cluster)
            for _ in range(n_clusters)
        ])

    def forward(self, cl_ids, pc_seq, dglob_seq):
        p = self.pc_embed(pc_seq)           # (B, L, pc_emb)
        d = self.delta_embed(dglob_seq)     # (B, L, delta_emb)
        x = torch.cat([p, d], dim=-1)       # (B, L, pc_emb+delta_emb)
        x = self.in_proj(x)                 # (B, L, dim)
        for blk in self.blocks:
            x = blk(x)                      # (B, L, dim)

        xT = x.transpose(1, 2)              # (B, dim, L)
        pooled = torch.cat([self.pool_avg(xT).squeeze(-1),
                            self.pool_max(xT).squeeze(-1)], dim=1)  # (B, 2*dim)
        pooled = self.drop(pooled)

        c = self.cluster_embed(cl_ids)      # (B, cluster_emb)
        h = torch.cat([pooled, c], dim=1)   # (B, 2*dim + cluster_emb)

        B = h.size(0)
        out = torch.empty(B, self.n_classes, device=h.device)
        for k in range(self.n_clusters):
            m = (cl_ids == k)
            if m.any():
                out[m] = self.heads[k](h[m])
        return out

# ───────────────────────────── Utilities ─────────────────────────────
def device_from_arg(s: str) -> torch.device:
    s = (s or "auto").lower()
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def atomic_write_lines(path: str, lines: List[str]) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_cnnpref_", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            for ln in lines:
                f.write(ln.rstrip() + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def bucketize_delta(delta: int, mag_bins: int) -> int:
    sign = 0 if delta >= 0 else 1
    mag = abs(int(delta))
    if mag == 0:
        m = 0
    else:
        m = int(math.floor(math.log2(mag)))
        m = max(0, min(m, mag_bins - 1))
    return sign * mag_bins + m

def parse_history_csv(path: str):
    """
    Expect lines: addr_hex,ip_hex,cache_hit,TYPE
    Ignore malformed / torn tail lines.
    Returns NumPy arrays:
      blocks (dtype=object), pcs (dtype=object), hits (int8)
    """
    blocks, pcs, hits = [], [], []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(",")
            if len(parts) < 3:
                continue
            a, p, h = parts[0].strip(), parts[1].strip(), parts[2].strip()
            try:
                addr = int(a, 16) if a.startswith("0x") else int(a, 16)
                ip   = int(p, 16) if p.startswith("0x") else int(p, 16)
                hit  = int(h)
            except Exception:
                continue
            blocks.append(addr)  # Python int (arbitrary precision)
            pcs.append(ip)      # Python int
            hits.append(hit)
    return np.array(blocks, dtype=object), np.array(pcs, dtype=object), np.array(hits, dtype=np.int8)

def assign_clusters_1d(blocks_obj: np.ndarray, centers_flat) -> np.ndarray:
    """
    blocks_obj: (N,) object array of Python ints
    centers_flat: iterable of centers (float)
    Returns cluster ids (int64) by nearest center in 1D (L1/L2 same in 1D up to sqrt).
    """
    centers = np.asarray(centers_flat, dtype=np.float64).reshape(-1)   # (C,)
    blocks_f = np.asarray(blocks_obj, dtype=np.float64).reshape(-1)    # (N,)
    if blocks_f.size == 0:
        return np.empty((0,), dtype=np.int64)
    dists = np.abs(centers[:, None] - blocks_f[None, :])               # (C, N)
    return dists.argmin(axis=0).astype(np.int64)

def build_delta_in_ids(deltas_list, cluster_id: int,
                       delta_out2id_per: Dict[int, Dict[int, int]],
                       top_per_cl: int,
                       use_tail_buckets: bool,
                       tail_mag_bins: int):
    """
    Map each Python-int delta to its input id (head id or bucket id).
    Returns a list of ints.
    """
    m = delta_out2id_per.get(int(cluster_id), {})
    out = []
    for d in deltas_list:
        d_int = int(d)
        if d_int in m:
            out.append(m[d_int])
        else:
            if use_tail_buckets:
                b = bucketize_delta(d_int, tail_mag_bins)
                out.append(top_per_cl + b)
            else:
                out.append(top_per_cl)  # UNK (not used if always using buckets)
    return out

@torch.no_grad()
def logits_to_topk_deltas(logits: torch.Tensor,
                          cl_id: int,
                          id2delta_per: Dict[int, Dict[int, int]],
                          bucket_fallbacks: Dict[int, Dict[int, List[int]]],
                          topk: int,
                          top_per_cl: int) -> List[int]:
    """
    Single-sample: map top classes to concrete deltas (expand buckets via fallbacks).
    """
    C = logits.shape[1]
    k_eff = min(topk, C)
    idx = logits.topk(k_eff, dim=1).indices[0].tolist()
    seen, out = set(), []
    for cls in idx:
        if cls < top_per_cl:
            d = id2delta_per.get(int(cl_id), {}).get(int(cls), None)
            if d is not None and d not in seen:
                out.append(int(d)); seen.add(int(d))
        else:
            b = int(cls) - top_per_cl
            for d in bucket_fallbacks.get(int(cl_id), {}).get(b, []):
                if d not in seen:
                    out.append(int(d)); seen.add(int(d))
                if len(out) >= topk:
                    break
        if len(out) >= topk:
            break
    return out[:topk]

# ───────────────────────────── Inference ─────────────────────────────
def run_inference(ckpt_path: str, in_csv: str, out_path: str,
                  topk: int = 10, history_len: int = 64,
                  device_str: str = "auto", block_bytes: int = 64):
    device = device_from_arg(device_str)

    # PyTorch 2.6: explicitly allow full pickle (trusted local file)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    # Config bits
    CLUSTERS    = int(cfg["CLUSTERS"])
    TOP_PER_CL  = int(cfg["TOP_PER_CL"])
    N_DELTA_IN  = int(cfg["n_delta_inputs_per_cluster"])
    N_CLASSES   = int(cfg["n_classes_per_cluster"])
    PC_EMB      = int(cfg["pc_emb"])
    DELTA_EMB   = int(cfg["delta_emb"])
    CL_EMB      = int(cfg["cluster_emb"])
    MIXER_DIM   = int(cfg["mixer_dim"])
    MIXER_DEPTH = int(cfg["mixer_depth"])
    TOK_DIM     = int(cfg["token_mlp_dim"])
    CHN_DIM     = int(cfg["channel_mlp_dim"])
    SEMI_BIN    = bool(cfg["semi_binary"])
    DROPOUT     = float(cfg["dropout"])
    USE_TAIL    = bool(cfg["use_tail_buckets"])
    MAG_BINS    = int(cfg["tail_mag_bins"]) if USE_TAIL else 0

    # Vocab/mappings + centers
    pc2id = {int(k): int(v) for k, v in ckpt["pc2id"].items()}
    id2delta_per = {int(c): {int(i): int(d) for i, d in m.items()} for c, m in ckpt["id2delta_per"].items()}
    delta_out2id_per = {int(c): {int(d): int(i) for d, i in m.items()} for c, m in ckpt["delta_out2id_per"].items()}
    bucket_fallbacks = {int(c): {int(b): [int(x) for x in arr] for b, arr in m.items()} for c, m in ckpt["bucket_fallbacks"].items()}
    centers_flat = np.asarray(ckpt["kmeans_centers"], dtype=np.float64).reshape(-1)

    # Rebuild model
    n_pcs_with_unk = int(cfg["n_pcs_with_unk"])
    model = PrefetchBiMixer(
        n_clusters=CLUSTERS,
        n_pcs_with_unk=n_pcs_with_unk,
        n_delta_inputs_per_cluster=N_DELTA_IN,
        pc_emb=PC_EMB, delta_emb=DELTA_EMB, cluster_emb=CL_EMB,
        dim=MIXER_DIM, mixer_depth=MIXER_DEPTH,
        token_mlp_dim=TOK_DIM, channel_mlp_dim=CHN_DIM,
        dropout=DROPOUT, n_classes_per_cluster=N_CLASSES,
        semi_binary=SEMI_BIN, seq_len=history_len
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    # Parse history CSV
    blocks_all, pcs_all, hits_all = parse_history_csv(in_csv)

    # Miss-only stream (matches training)
    miss_mask = (hits_all == 0)
    if miss_mask.sum() == 0:
        atomic_write_lines(out_path, [])
        return

    blocks = blocks_all[miss_mask]
    pcs    = pcs_all[miss_mask]

    # (Optional) align blocks to cache line if needed (training claims already aligned)
    if block_bytes and (block_bytes & (block_bytes - 1)) == 0:
        mask = ~((block_bytes) - 1) & ((1 << 64) - 1)
        blocks = np.array([int(b) & mask for b in blocks.tolist()], dtype=object)

    # Cluster assignment via saved centers
    cls = assign_clusters_1d(blocks, centers_flat)
    if cls.size == 0:
        atomic_write_lines(out_path, [])
        return
    cl_curr = int(cls[-1])  # cluster of the last miss

    # Indices of this cluster; need last `history_len` samples
    idx_c = np.nonzero(cls == cl_curr)[0]
    
    # If the cluster never appeared, we truly cannot build a window
    if len(idx_c) == 0:
        atomic_write_lines(out_path, [])
        # optional: debug_log(out_path, [f"no_samples_for_cluster {cl_curr}"])
        return
    
    # Take what we have for this cluster
    take_idx = idx_c[-history_len:]                  # may be shorter than history_len
    seq_blocks_list = [int(blocks[j]) for j in take_idx]
    seq_pcs_list    = [int(pcs[j])    for j in take_idx]
    
    # Left-pad to full length if needed
    if len(seq_blocks_list) < history_len:
        need = history_len - len(seq_blocks_list)
        pad_block = seq_blocks_list[0]               # repeat first block → 0 deltas
        seq_blocks_list = [pad_block]*need + seq_blocks_list
        seq_pcs_list    = [0]*need + seq_pcs_list    # PC UNK id (0) will be applied below
    
    # Convert to arrays for the rest of the pipeline
    seq_blocks = np.array(seq_blocks_list, dtype=object)
    seq_pcs    = np.array(seq_pcs_list,    dtype=object)

    # Local deltas in Python-int space
    seq_blocks_py = [int(b) for b in seq_blocks.tolist()]
    local_deltas = [0] + [seq_blocks_py[i] - seq_blocks_py[i-1] for i in range(1, len(seq_blocks_py))]

    # Map to ids (inputs)
    pc_ids = np.array([pc2id.get(int(x), 0) for x in seq_pcs.tolist()], dtype=np.int64)
    delta_in_ids = build_delta_in_ids(local_deltas, cl_curr, delta_out2id_per, TOP_PER_CL, USE_TAIL, MAG_BINS)
    dglob = (cl_curr * N_DELTA_IN) + np.asarray(delta_in_ids, dtype=np.int64)

    # Tensors
    cl_t = torch.tensor([cl_curr], dtype=torch.long, device=device)          # (1,)
    pc_t = torch.tensor(pc_ids[None, :], dtype=torch.long, device=device)    # (1, L)
    dg_t = torch.tensor(dglob[None, :], dtype=torch.long, device=device)     # (1, L)

    with torch.no_grad():
        logits = model(cl_t, pc_t, dg_t)  # (1, C)

    # Map logits → deltas → absolute blocks (wrap to 64-bit)
    cand_deltas = logits_to_topk_deltas(logits, cl_curr, id2delta_per, bucket_fallbacks, topk, TOP_PER_CL)
    last_block  = seq_blocks_py[-1]
    
    PREF_MAX_BITS = 48          # typical VA width; tune if your traces differ
    PREF_UPPER    = (1 << PREF_MAX_BITS) - 1
    
    pred_blocks, seen = [], set()

    for d in cand_deltas:
        b_signed = last_block + int(d)
        if b_signed < 0:
            continue                        # drop underflow
        if b_signed > PREF_UPPER:
            continue                        # drop implausible high addrs
        # (optional) also drop huge leaps:
        # if abs(int(d)) > (1 << 20):  # ~64MB for 64B lines
        #     continue
        if b_signed not in seen:
            pred_blocks.append(b_signed)
            seen.add(b_signed)

    # Emit hex (one per line)
    hex_lines = [f"0x{b:x}" for b in pred_blocks]
    debug_lines = [
        f"# cluster {cl_curr}",
        f"# cwd={os.getcwd()}",
        f"# in={in_csv}",
        f"# out={out_path}",
        f"# ckpt={ckpt_path}",
    ] + hex_lines
    debug_path = "debug.txt"   
    with open(debug_path, "w") as df:
        df.write("# Debug info for prefetch inference\n")
        for ln in debug_lines:
            df.write(ln + "\n")

    atomic_write_lines(out_path, hex_lines)

# ──────────────────────────────── CLI ───────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt_path = os.path.join(script_dir, "files", "weights")
    ckpt_file_list = [f for f in os.listdir(default_ckpt_path) if f.endswith('.pt')]
    if not ckpt_file_list:
        raise FileNotFoundError(f"No checkpoint file found in {default_ckpt_path}")
    default_ckpt = os.path.join(default_ckpt_path, ckpt_file_list[0])
    ap = argparse.ArgumentParser(description="BiMixer prefetch inference → top-K addresses")
    ap.add_argument("--ckpt", type=str, help="Path to checkpoint (prefetch_mixer_ckpt_best.pt)", default=default_ckpt)
    ap.add_argument("--input_file", required=True, type=str, help="Path to history CSV (addr,ip,cache_hit,TYPE)")
    ap.add_argument("--output_file", required=True, type=str, help="Where to write predictions (one hex per line)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--history_len", type=int, default=64)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument("--block_bytes", type=int, default=64, help="Cache line bytes; 0 to skip alignment")
    args = ap.parse_args()

    run_inference(
        ckpt_path=args.ckpt,
        in_csv=args.input_file,
        out_path=args.output_file,
        topk=args.topk,
        history_len=args.history_len,
        device_str=args.device,
        block_bytes=args.block_bytes,
    )

if __name__ == "__main__":
    main()

