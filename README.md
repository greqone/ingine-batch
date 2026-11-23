# ingine-batch

Many concurrent requests to [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on one
RTX 5090, decoded together.

ingine-batch is a fork of **ingine**, a C++/CUDA inference engine for Inkling-2-Mini
([satellitedown/inkling-2-mini-abliterated-ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine)),
that adds batched serving: up to N requests share every decode step, with batched DSpark speculative decoding. It
serves the original model with the checkpoint's weights as stored (the routed experts in NVFP4 and MXFP8, everything
else in BF16) and BF16 KV caches; nothing is quantized further, and its outputs pass the same validation against
the reference implementation as single-request ingine's (see [Correctness](#correctness)).

## Results

RTX 5090 (32 GB), `ingine-serve --batch 64 --batch-ctx 640 --batch-spec 4 --batch-spec-threshold 0.5
--batch-spec-min 2`, 64 concurrent streaming requests of short prompts
([bench/prompts-short.json](bench/prompts-short.json)), at most 512 output tokens each, the model's default thinking
effort, measured over HTTP with [bench/concurrent.py](bench/concurrent.py) (one warm-up request, then three rounds;
median, and the range of the rounds):

| Sampling | Total output tok/s (wall clock) | Sum of the requests' tok/s |
| --- | ---: | ---: |
| greedy | 4,176 (4,174 to 4,219) | 5,422 |
| temperature 1 (the model's default) | 3,507 (3,505 to 3,702) | 4,586 |

- **Total output tok/s (wall clock):** the completion tokens of all 64 requests divided by the time from the first
  request sent to the last response ended, as the "concurrent aggregate" of
  [Strata's benchmark protocol](https://github.com/Niko1221/Strata/blob/main/docs/BENCHMARK_PROTOCOL.md). It
  includes admission (the 64 prompts are read in about 0.3 s) and the round's tail, when the requests that accepted
  fewer drafts are still running.
- **Sum of the requests' tok/s:** each request's (completion tokens - 1) / (last token - first token), summed: the
  other way concurrent throughput is quoted ("the streams' tok/s added").

The GPU ran with the NVIDIA application profile in [tools/50-ingine-batch](tools/50-ingine-batch), which lets
`ingine-serve` use the P0 performance state (see [Serve](#serve)); without it, the greedy rounds measured 4,183 to
4,229 tok/s (median 4,200), the same within the rounds' spread.

For comparison, Strata v0.2.19 reports **4,089.7 tok/s** "total output across 64 concurrent requests on RTX 5090"
for Inkling-2-Mini ([release notes](https://github.com/Niko1221/Strata/releases/tag/v0.2.19)); its prompts,
output lengths, sampling, quantization and metric for that figure are not published. With greedy decoding, all six
rounds (with and without the profile) are higher on the wall-clock total, 4,174 to 4,229 tok/s, and on the summed
rate; at temperature 1, the summed rate is higher (4,586) and the wall-clock total lower (3,507).

## How it works

- **Slots.** Each request gets a slot: its own sliding-window rings (40 attention visits, min(ctx, 1024) rows, which
  never wrap when the slot holds fewer than 1024 positions), full-attention caches (8 visits) and short-convolution
  states. `--batch-ctx` sets the positions per slot (prompt, output and the rows of one verify step).
- **A step** runs the model once for every running request: the same visit sequence as the prefill (48 visits of 24
  layers; layers 6-17 run three times) over all requests' rows, each request with its own number of rows (1 to 8).
  Dense projections are BF16 GEMMs (cuBLASLt, the algorithm per shape chosen by timing, and an own kernel for the
  head, which also picks every row's token); the routed experts are read once per step for all rows
  (`src/kernels/batch/moe.cu`: grouped tiles over the touched experts, the NVFP4 and MXFP8 weights decoded exactly
  in registers); attention is one kernel over all requests, each against its own slot
  (`src/kernels/attn/attn_batch.cu`).
- **Speculative decoding** (`--batch-spec K`): the batched DSpark drafter (`src/model/batch_drafter.*`, the
  checkpoint's `mtp.safetensors`) proposes up to K tokens per request; one step verifies [pending token, its drafts]
  of every request. With `--batch-spec-threshold`, a request verifies the drafts whose estimated acceptance (the
  product of the drafter's confidences) exceeds the threshold, scaled by the share of slots still running, but at
  least `--batch-spec-min` of them. A request keeps the leading drafts that equal the tokens the target itself
  samples at those rows (its argmax when greedy), then the target's next token. Every emitted token is the target's
  own sample, so the output is distributed exactly as plain decoding at every temperature. The convolution states
  are written once per step, from each request's kept rows; the caches need no rollback, since rows past a
  request's length are never read before they are written again.
- **Admission.** Prompts of up to 256 tokens are read together, up to 8 tokens of each prompt per step; longer ones
  go through the single-request prefill one after the other.
- **Scheduling.** `engine::BatchScheduler` admits waiting requests into free slots, runs one step over every running
  request, streams each its tokens and frees the slots of requests that end (stop token, `max_tokens`, slot full,
  client gone). The HTTP layer, chat template, streaming and tool-call parsing are ingine's
  (`/v1/chat/completions`, `/v1/models`, `/health`).

Without `--batch`, `ingine-serve` serves one request at a time as ingine does (with the same validation results).

## Correctness

`ingine-validate` compares the engine's logits with goldens of the Python reference implementation for the original
model (a 30-token chat prompt and a 2,100-token prompt, prefilled in the goldens' chunks, then 32 greedy tokens
teacher forced; KL divergence over the reference's top 256 tokens and top-1 agreement; `--cases orig_`). `--batch`
runs the goldens through the batched path, all cases at once in separate slots; `--width M` verifies them in steps
of M rows, of which each case keeps 1 to M (a different count every step) and fills the rest with wrong tokens,
rejected and rolled back, so every compared row comes after rollbacks; `--ragged` gives each case its own row count
(1 to M) in every step, as speculative decoding does:

| Path | Stages | Top-1 agreement | KL mean | KL max |
| --- | ---: | ---: | ---: | ---: |
| single-request decode (ingine) | 66 | 0.985 | 1.00e-3 | 0.016 |
| batched, 1 row per request | 66 | 0.970 | 1.68e-3 | 0.050 |
| batched, 1-4 rows per request, with rollbacks | 66 | 0.955 | 2.38e-3 | 0.096 |
| batched, 1-8 rows per request, with rollbacks | 66 | 0.970 | 2.63e-3 | 0.104 |

ingine's full golden set (330 stages; it adds the abliterated variant of the model, with BF16 and FP8
full-attention caches) passes on every path as well, with KL means of 2.3e-3 to 2.9e-3.

On long prompts the reference already differs from itself by KL 6e-3 to 8e-2 per stage between two valid
configurations (prefill chunking, attention precision); a real defect shows as KL above 1. Teacher forced on
single-request ingine's greedy outputs for the 64 benchmark prompts (31,774 positions; `ingine-batch-bench
--forced`), the batched engine picks the same token at 98.9% of them; where it does not, it scores single-request
ingine's token 0.06 logits below its own choice at the median and 0.83 at most, so the two engines' free-running
greedy outputs part at the first such position (after 74 tokens at the median).

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
  --batch 64 --batch-ctx 640 --batch-spec 4 --batch-spec-threshold 0.5 --batch-spec-min 2 --max-queue 64
```

- `--batch N`: requests decoded together; more wait in a queue of `--max-queue`.
- `--batch-ctx C`: positions per slot (prompt + `max_tokens` + K). The slots' caches take about 0.13 GB each at 640
  positions; 64 slots of 640 positions, with the drafter and scratch, take 10.2 GB and leave about 2.3 GB of the
  32 GB. Fewer slots allow longer contexts (32 slots of 2,048 positions take about 7.5 GB of caches).
- `--batch-spec K`: the most drafts a request verifies per step (0 turns speculative decoding off).
- `--batch-spec-threshold θ`: per request and step, the leading drafts whose estimated acceptance (the product of
  the drafter's confidences) exceeds θ x (busy slots / slots); 0 verifies K drafts every step.
- `--batch-spec-min k`: with a threshold, at least k drafts per request.

The settings above are the fastest of those measured at 64 concurrent requests.

It speaks the OpenAI chat completions API on `http://127.0.0.1:8090/v1` (streaming, `reasoning_effort`,
`stream_options.include_usage`). Batched serving takes text only, samples by temperature (no `top_p`/`top_k`) and
loads the dense weights in BF16 (`--dense-fp8` is single-request only).

**GPU performance state.** The NVIDIA driver keeps CUDA processes in the P1 performance state by default (memory
clock 13,801 MHz on the RTX 5090). The application profile [tools/50-ingine-batch](tools/50-ingine-batch) sets the
driver's CudaNoStablePerfLimit for `ingine-serve` and `ingine-batch-bench`, so they can reach P0 (14,001 MHz), for
processes started after it is installed (driver 595 or newer):

```bash
mkdir -p ~/.nv/nvidia-application-profiles-rc.d && cp tools/50-ingine-batch ~/.nv/nvidia-application-profiles-rc.d/
```

## Benchmark

```bash
python3 bench/concurrent.py --port 8090 --concurrency 64 --max-tokens 512 --rounds 3             # default sampling
python3 bench/concurrent.py --port 8090 --concurrency 64 --max-tokens 512 --rounds 3 --temperature 0
```

`native/build/apps/batch/ingine-batch-bench` measures the engine without HTTP (`--seqs 64 --tokens 512 --ctx 640
--effort high --spec 4 --spec-threshold 0.5 --spec-min 2`; `high` is the chat template's default effort; `--help`
lists the test modes).

## License

Apache License 2.0 ([LICENSE](LICENSE)); vendored dependencies keep their own licenses, see [NOTICE](NOTICE).
Inkling-2-Mini is by Thinking Machines (Apache License 2.0). This is an unofficial community project.
