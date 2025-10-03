
python3 prefetcher/bnnpref/model/infer_server.py \
  --sock /tmp/bnnpref.sock \
  --ckpt prefetcher/bnnpref/model/files/weights/prefetch_mixer_ckpt_p10_93.16_p1_62.93.pt \
  --device auto \
  --threads 1 \
  --max-req 500000000 \
  --progress-label "bnnpref"
