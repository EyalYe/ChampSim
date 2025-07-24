import json
import random

VOCAB_PATH = "vocab.json"
HISTORY_LENGTH = 16

with open(VOCAB_PATH) as f:
    vocab = json.load(f)

# Extract keys (and convert back to int)
pc_keys = [int(k) for k in vocab['pc2id'].keys()]
delta_keys = [int(k) for k in vocab['delta2id'].keys()]

# Randomly sample HISTORY_LENGTH PCs and deltas
pcs = random.choices(pc_keys, k=HISTORY_LENGTH)
deltas = random.choices(delta_keys, k=HISTORY_LENGTH)

# Format as required: "pc1,pc2,...;delta1,delta2,..."
pc_str = ','.join(str(p) for p in pcs)
delta_str = ','.join(str(d) for d in deltas)

# Output the string to be passed to the inference script
print(f"{pc_str};{delta_str}")

