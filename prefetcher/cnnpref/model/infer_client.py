#!/usr/bin/env python3
import os, sys, json, socket, argparse

def send(sock_path, obj):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall(json.dumps(obj).encode("utf-8"))
    s.shutdown(socket.SHUT_WR)
    data = b""
    while True:
        chunk = s.recv(65536)
        if not chunk: break
        data += chunk
    s.close()
    return json.loads(data.decode("utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/cnnpref.sock")
    ap.add_argument("--in", dest="inf", required=False, help="history CSV")
    ap.add_argument("--out", dest="outf", required=False, help="predictions txt")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--hist", type=int, default=64)
    ap.add_argument("--block_bytes", type=int, default=64)
    ap.add_argument("--ping", action="store_true")
    args = ap.parse_args()

    if args.ping:
        print(send(args.sock, {"cmd":"PING"})); return

    if not args.inf or not args.outf:
        ap.error("Provide --in and --out (or use --ping)")

    req = {"in": args.inf, "out": args.outf, "topk": args.topk, "hist": args.hist, "block_bytes": args.block_bytes}
    resp = send(args.sock, req)
    print(resp)

if __name__ == "__main__":
    main()

