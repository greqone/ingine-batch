# ingine-batch

Many concurrent requests to [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on one
RTX 5090, decoded together.

ingine-batch is a fork of **ingine**, a C++/CUDA inference engine for Inkling-2-Mini
([satellitedown/inkling-2-mini-abliterated-ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine)),
that adds batched serving: up to N requests share every decode step, with batched DSpark speculative decoding. It
serves the original model with the checkpoint's weights as stored (the routed experts in NVFP4 and MXFP8, everything
else in BF16) and BF16 KV caches; nothing is quantized further, and its outputs agree with the reference
implementation as closely as single-request ingine's do (see [Correctness](#correctness)).

## Results

RTX 5090 (32 GB), `ingine-serve --batch 64 --batch-ctx 640 --batch-spec 2`, 64 concurrent streaming requests of
short prompts ([bench/prompts-short.json](bench/prompts-short.json)), at most 512 output tokens each, the model's
default thinking effort, measured over HTTP with [bench/concurrent.py](bench/concurrent.py) (one warm-up request,
then three rounds; median):

| Sampling | Total output tok/s (wall clock) | Sum of the requests' tok/s |
| --- | ---: | ---: |
| greedy | 3,493 | 4,393 |
| temperature 1 (the model's default) | 2,987 | 3,908 |

- **Total output tok/s (wall clock):** the completion tokens of all 64 requests divided by the time from the first
  request sent to the last response ended, as the "concurrent aggregate" of
  [Strata's benchmark protocol](https://github.com/Niko1221/Strata). It includes admission (the 64 prompts are read
  in about 0.3 s) and the round's tail, when the requests that accepted fewer drafts are still running.
- **Sum of the requests' tok/s:** each request's (completion tokens - 1) / (last token - first token), summed: the
  other way concurrent throughput is quoted ("the streams' tok/s added").

For comparison, Strata v0.2.19 reports **4,089.7 tok/s** "total output across 64 concurrent requests on RTX 5090"
for Inkling-2-Mini ([release notes](https://github.com/Niko1221/Strata/releases/tag/v0.2.19)); its prompts,
output lengths, sampling, quantization and metric for that figure are not published.

## How it works

- **Slots.** Each request gets a slot: its own sliding-window rings (40 attention visits, min(ctx, 1024) rows, which
  never wrap when the slot holds fewer than 1024 positions), full-attention caches (8 visits) and short-convolution
  states. `--batch-ctx` sets the positions per slot (prompt, output and the rows of one verify step).
- **A step** runs the model once for every running request: the same visit sequence as the prefill (48 visits of 24
  layers; layers 6-17 run three times), over 64 x (K + 1) rows. Dense projections are BF16 cuBLASLt GEMMs; the
  routed experts are read once per step for all rows (`src/kernels/batch/moe.cu`: grouped tiles over the touched
  experts, the NVFP4 and MXFP8 weights decoded exactly in registers); attention is one kernel over all sequences,
  each against its own slot (`src/kernels/attn/attn_batch.cu`).
- **Speculative decoding** (`--batch-spec K`): the batched DSpark drafter (`src/model/batch_drafter.*`, the
  checkpoint's `mtp.safetensors`) proposes K tokens per request; one step verifies [pending token, K drafts] of
  every request. A request keeps the leading drafts that equal the tokens the target itself samples at those rows
  (its argmax when greedy), then the target's next token. Every emitted token is the target's own sample, so the
  output is distributed exactly as plain decoding at every temperature. Rejected rows are rolled back: each slot's
  convolution states return to their state after the kept rows (the overwritten rows are saved before the step);
  the caches need no rollback, since rows past a request's length are never read before they are written again.
- **Admission.** Prompts of up to 256 tokens are read together, in steps of up to 8 rows per prompt; longer ones go
  through the single-request prefill one after the other.
- **Scheduling.** `engine::BatchScheduler` admits waiting requests into free slots, runs one step over every running
  request, streams each its tokens and frees the slots of requests that end (stop token, `max_tokens`, slot full,
  client gone). The HTTP layer, chat template, streaming and tool-call parsing are ingine's
  (`/v1/chat/completions`, `/v1/models`, `/health`).

The single-request engine is unchanged; `ingine-serve` without `--batch` serves one request at a time as before.

## Correctness

`ingine-validate` compares the engine's logits with goldens of the Python reference implementation (prompts of 30
to 5,000 tokens, then 32 greedy tokens teacher forced; KL divergence over the reference's top 256 tokens and top-1
agreement). `--batch` runs the goldens through the batched path, all cases at once in separate slots; `--width M`
verifies them in steps of M rows, of which each case keeps 1 to M (a different count every step) and fills the rest
with wrong tokens, rolled back after the step, so every compared row comes after rollbacks:

| Path | Stages | Top-1 agreement | KL mean | KL max |
| --- | ---: | ---: | ---: | ---: |
| single-request decode (ingine) | 330 | 0.979 | 2.32e-3 | 0.105 |
| batched, 1 row per request | 330 | 0.976 | 2.26e-3 | 0.113 |
| batched, 4 rows per request, with rollbacks | 330 | 0.976 | 2.43e-3 | 0.093 |
| batched, 8 rows per request, with rollbacks | 330 | 0.973 | 2.25e-3 | 0.117 |

On these long prompts the reference already differs from itself by KL 6e-3 to 8e-2 per stage between two valid
configurations (prefill chunking, attention precision); a real defect shows as KL above 1. Greedy outputs of the
batched engine, with and without speculative decoding, match single-request ingine token for token until the
first near tie (logit gaps of a few tenths at most), the same as two valid kernel configurations of one engine.

## Requirements

Linux x86_64, an NVIDIA GPU with compute capability 12.0 and 32 GB (RTX 5090; the RTX PRO Blackwell cards are also
compute capability 12.0 but untested), an NVIDIA driver for CUDA 13, the CUDA 13 toolkit (nvcc, cuBLAS), CMake 3.28+,
Ninja and a C++20 compiler (GCC 12 to 16). About 21 GB of disk for the model.

## Build

```bash
git clone https://github.com/satellitedown/ingine-batch
cd ingine-batch/native
cmake --preset release -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc   # the nvcc of your CUDA 13 toolkit
cmake --build build -j
```

The server is `native/build/apps/serve/ingine-serve`. `--preset dev` also builds the tests (`ctest --test-dir build`;
the GPU tests need the model, `INGINE_MODEL_DIR`).

## Model

The original checkpoint from Hugging Face (19.5 GB of weights and the 0.6 GB DSpark drafter, used as stored; `hf` is
the CLI of `pip install -U huggingface_hub`):

```bash
hf download thinkingmachines/Inkling-2-Mini --revision b85c3cfa615fc413cc9af3379f491b963f3b586a \
  --local-dir ~/models/Inkling-2-Mini
```

## Serve

```bash
native/build/apps/serve/ingine-serve --engine --model-dir ~/models/Inkling-2-Mini --port 8090 \
  --batch 64 --batch-ctx 640 --batch-spec 2 --max-queue 64
```

- `--batch N`: requests decoded together; more wait in a queue of `--max-queue`.
- `--batch-ctx C`: positions per slot (prompt + `max_tokens` + K). The slots' caches take about 0.13 GB each at 640
  positions; 64 slots of 640 positions leave about 1.5 GB of the 32 GB. Fewer slots allow longer contexts (32 slots
  of 2,048 positions take about 7.5 GB).
- `--batch-spec K`: drafts verified per step (2 is fastest at 64 requests; 0 turns speculative decoding off).

It speaks the OpenAI chat completions API on `http://127.0.0.1:8090/v1` (streaming, `reasoning_effort`,
`stream_options.include_usage`). Batched serving takes text only, samples by temperature (no `top_p`/`top_k`) and
loads the dense weights in BF16 (`--dense-fp8` is single-request only).

## Benchmark

```bash
python3 bench/concurrent.py --port 8090 --concurrency 64 --max-tokens 512 --rounds 3             # default sampling
python3 bench/concurrent.py --port 8090 --concurrency 64 --max-tokens 512 --rounds 3 --temperature 0
```

`native/build/apps/batch/ingine-batch-bench` measures the engine without HTTP (`--seqs 64 --tokens 512 --ctx 640
--effort high --spec 2`; `--help` lists the test modes).

## License

Apache License 2.0 ([LICENSE](LICENSE)); vendored dependencies keep their own licenses, see [NOTICE](NOTICE).
Inkling-2-Mini is by Thinking Machines (Apache License 2.0). This is an unofficial community project.
