# infr_single.py
import sys
import torch
import torch.nn.functional as F
import json
from model import PrefetchCNN

MODEL_PATH = "prefetch_cnn_weights.pth"
VOCAB_PATH = "vocab.json"
HISTORY_LENGTH = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Read input: "pc1,pc2,...,pc16;d1,d2,...,d16"
    line = sys.stdin.read().strip()
    pcs_str, deltas_str = line.split(';')
    pcs = list(map(int, pcs_str.split(',')))
    deltas = list(map(int, deltas_str.split(',')))

    # Load vocab
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    pc2id = {int(k): v for k, v in vocab["pc2id"].items()}
    delta2id = {int(k): v for k, v in vocab["delta2id"].items()}
    id2delta = {int(k): v for k, v in vocab["id2delta"].items()}

    n_pcs = vocab["n_pcs"]
    n_deltas = vocab["n_deltas"]

    # Convert real PCs/deltas to IDs
    try:
        pc_ids = [pc2id[pc] for pc in pcs]
        delta_ids = [delta2id[d] for d in deltas]
    except KeyError as e:
        print(f"Unknown value: {e}", file=sys.stderr)
        return

    # Load model
    model = PrefetchCNN(n_pcs, n_deltas).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    pc_tensor = torch.LongTensor([pc_ids]).to(device)
    delta_tensor = torch.LongTensor([delta_ids]).to(device)

    with torch.no_grad():
        logits = model(pc_tensor, delta_tensor)
        pred_id = torch.argmax(logits, dim=1).item()
        pred_delta = id2delta[pred_id]

    print(pred_delta, pred_id)

if __name__ == "__main__":
    main()

