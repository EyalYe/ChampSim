
python3 prefetcher/cnnpref/model/infer_server.py \
  --ckpt $1 \
  --sock /tmp/cnnpref.sock --history_len 64 --device cpu --threads 1 \
  --device cuda

