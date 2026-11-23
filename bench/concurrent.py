#!/usr/bin/env python3
"""Concurrent output throughput of an OpenAI-compatible chat completions server.

A round releases N streaming requests together, each on its own connection, then waits for all of them. The round's
aggregate is the sum of the completion tokens the server reports (stream_options.include_usage) over the wall time
from the first request sent to the last response ended: "total output tokens/s across N concurrent requests".
One warm-up request runs before the rounds and is not counted.

    python3 bench/concurrent.py --port 8090 --concurrency 64 --max-tokens 512 --rounds 3
"""

import argparse
import http.client
import json
import pathlib
import statistics
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent


def request(args, prompt, i):
    """One streaming chat completion; returns (t_sent, t_first_token, t_end, completion_tokens, text)."""
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.temperature is not None:
        body["temperature"] = args.temperature
    if args.reasoning_effort is not None:
        body["reasoning_effort"] = args.reasoning_effort
    conn = http.client.HTTPConnection(args.host, args.port, timeout=args.timeout)
    t_sent = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"request {i}: HTTP {resp.status}: {resp.read()[:300]!r}")
    t_first, tokens, text = None, None, []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                if t_first is None:
                    t_first = time.perf_counter()
                text.append(piece)
        if chunk.get("usage"):
            tokens = chunk["usage"]["completion_tokens"]
    t_end = time.perf_counter()
    conn.close()
    if tokens is None:
        raise RuntimeError(f"request {i}: no usage in the stream")
    return t_sent, t_first or t_end, t_end, tokens, "".join(text)


def run_round(args, prompts):
    results = [None] * args.concurrency
    errors = []
    start = threading.Barrier(args.concurrency)

    def worker(i):
        try:
            start.wait()
            results[i] = request(args, prompts[i % len(prompts)], i)
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(f"{e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError("; ".join(errors[:3]))
    t0 = min(r[0] for r in results)
    t1 = max(r[2] for r in results)
    total = sum(r[3] for r in results)
    ttft = [r[1] - r[0] for r in results]
    per_request = [r[3] / (r[2] - r[1]) for r in results if r[2] > r[1] and r[3] > 1]
    capped = sum(r[3] >= args.max_tokens for r in results)
    return {
        "output_tokens": total,
        "wall_s": t1 - t0,
        "aggregate_tps": total / (t1 - t0),
        "ttft_p50_s": statistics.median(ttft),
        "ttft_max_s": max(ttft),
        "request_tps_p50": statistics.median(per_request) if per_request else 0.0,
        "capped": capped,
        "texts": [r[4] for r in results],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--prompts", default=str(ROOT / "prompts-short.json"), help="JSON array of prompt strings")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=None, help="omitted: the server's default")
    ap.add_argument("--reasoning-effort", default=None, help="omitted: the chat template's default")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default=None, help="write the rounds (with every response text) as JSON")
    args = ap.parse_args()

    prompts = json.loads(pathlib.Path(args.prompts).read_text())
    warm = request(args, prompts[0], -1)
    print(f"warm-up: {warm[3]} tokens in {warm[2] - warm[0]:.2f}s")
    rounds = []
    for k in range(args.rounds):
        r = run_round(args, prompts)
        rounds.append(r)
        print(f"round {k + 1}: {r['output_tokens']} output tokens in {r['wall_s']:.2f}s = {r['aggregate_tps']:.1f} tok/s "
              f"({r['capped']}/{args.concurrency} at max_tokens; TTFT p50 {r['ttft_p50_s']:.2f}s max "
              f"{r['ttft_max_s']:.2f}s; per request p50 {r['request_tps_p50']:.1f} tok/s)")
    agg = statistics.median(r["aggregate_tps"] for r in rounds)
    print(f"MEDIAN AGGREGATE {agg:.1f} tok/s over {args.rounds} rounds of {args.concurrency} concurrent requests")
    if args.out:
        settings = {k: v for k, v in vars(args).items() if k != "out"}
        pathlib.Path(args.out).write_text(json.dumps({"settings": settings, "rounds": rounds}, indent=1) + "\n")


if __name__ == "__main__":
    main()
