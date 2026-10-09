# Runtime optimization without changing the training recipe

These changes keep the draft architecture, parameter names, loss definitions,
optimizer/scheduler settings, dataset ordering, and acceptance-rate definitions.
They do not change the Qwen or DSV4 training scripts' batch sizes, worker counts,
prefetch settings, or recomputation policy.

## Step timings are independent of metric logging

`--log-freq` still controls training metrics. Synchronized step profiling is now
controlled by `--profile-freq`, with a default of **0 (disabled)**. This removes
profiling-specific accelerator synchronizations from ordinary steps; necessary
training synchronizations and metric reductions still run.

For diagnosis, add `--profile-freq 100` to the training command. Step zero and
every 100th optimizer step are sampled, independently of `--log-freq`. Profiling
changes execution timing and should not be used to measure uninstrumented peak
throughput.

`scripts/benchmark.py` explicitly enables profiling every step, since it needs a
complete set of warmup and measured samples. Its timings remain instrumented.

The existing `fetch_ms`, `fwd_ms`, `bwd_ms`, `opt_ms`, `step_ms`, `tokens_per_s`,
and `fetch_frac` keys are retained on sampled steps. Fetch timing no longer
includes the preceding step's metric logging or checkpoint save. Additional
fields partition the critical path:

| Field | Measured interval |
| --- | --- |
| `queue_wait_ms` | Waiting for the next DataLoader batch |
| `recovery_ms` | Batch recovery coordination |
| `h2d_ms` | Batch transfer to the accelerator |
| `model_fwd_ms` | Model forward, including model-side losses/metrics |
| `supervision_ms` | Trainer supervision accounting/reduction after forward |

`fetch_ms` covers queue, recovery, and transfer; `fwd_ms` covers model forward
and supervision accounting. Backward includes gradient clipping; optimizer
includes scheduler stepping. Distributed profiling adds `*_ms_rank_p50`,
`*_ms_rank_p95`, and `*_ms_rank_max` with one extra collective **only on sampled
steps**. Unsampled steps do not emit fake zero timings. The `step_ms` interval
excludes reporting, checkpointing, and profiling's rank-summary collective.

## Locate slow hidden-state requests

Set this environment variable on the trainer before launching its workers:

```bash
export SPECULATORS_HS_PROFILE_EVERY=100
```

The default is 0 (off). Each worker samples its first Arrow row and every Nth row
after that, without using random numbers or changing batch contents. One JSON
line per sampled row is written to stderr, identified by `event=hs_profile`,
`rank`, and `pid`. It reports request, cache read, generated-payload retrieval,
validation, cleanup/cache publication, and total row time. HTTP adds route,
download, and tensor-load timing where applicable, plus payload byte counts.
Nested HTTP stages are included in retrieval time: **do not sum them twice**.

| Worker field | Interpretation |
| --- | --- |
| `request_ms` | Target completion round trip, including server queue/prefill/HS publication and client retries; not HTTP transport alone |
| `get_generated_ms` | Retrieval of the returned HS artifact; includes the HTTP stages below |
| `http_route_ms` / `http_download_ms` / `http_load_ms` | Find the owning target, download the artifact, and load its tensors |
| `validate_ms` | Token/shape checks and finite-value scans before training |
| `cache_read_ms` / `cache_write_ms` / `delete_ms` | Cache lookup, cache publication, and generated-artifact cleanup |
| `tensor_bytes` / `http_file_bytes` | Logical bytes in the validated training prefix / complete downloaded file size |

`sample_total_ms` includes first-use setup and other row processing as well as
the named stages. Rows from prefetched workers overlap; adding their durations
does not give training wall time or pure target compute time.

The p50/p95 window is bounded to the latest 128 samples per stage; `max` and
`count` cover the worker's lifetime. Results describe sampled rows, not every
request. Paths, prompts, tokens, authentication secrets, and exception contents
are not logged. A slow `queue_wait_ms` alone does not prove HTTP is the cause;
compare worker request, download, load, and validation timings.

## Equivalent memory and data-path changes

- Large no-gradient probability tensors are released after their last use,
  before loss computation where possible. Loss arithmetic and kernels are
  unchanged; actual peak-memory savings require a hardware measurement.
- Compact model logging avoids confidence diagnostics it would discard anyway.
  Confidence BCE, enabled length losses, acceptance metrics, and gradients remain.
- Internally generated, already-validated HS do not undergo the same validation
  twice in one read. Cache reads and full multimodal responses remain validated;
  a remote/cache-supplied marker does not bypass validation.
- Mid-epoch resume uses the sampler's public `set_start_batch(epoch, index)` API.
  The full batch schedule stays intact, and skipped batches are not fetched.
  Starting a new epoch resets the offset. Existing RNG restoration is retained.

## Optional selective recomputation

The default `--activation-checkpointing` behavior still recomputes **all** draft
decoder layers. To try a subset explicitly:

```text
--activation-checkpointing --activation-checkpoint-layers 0 2
```

Correction and other heads are unaffected. Indices are zero-based; unset means
all layers, and an empty list in YAML means no selected layers. The option
requires checkpointing to be enabled and a supported dense backbone. It is a
runtime setting recorded in run provenance, not an architecture/checkpoint-tensor
change. Fewer recomputed layers use more activation memory. Keep the default if
the model only fits with full recomputation; do not disable it blindly.

HTTP spool limits and the opt-in Mooncake buffer/event settings are documented
in [HS transport tuning](dsv4_hs_transport_tuning.md). Mooncake pooling and copy
events remain disabled unless explicitly requested.

## Offline evaluation and validation limits

The existing offline evaluation script and import names remain supported. CLI,
loading, and orchestration are separated from decoding; sampling, rejection,
rollback, and acceptance statistics are unchanged.

CPU regressions cover fixed-input loss/gradient/update parity, RNG and resume
ordering, HS validation failures, and transfer lifetimes. They are not an A3
throughput or precision benchmark. Before changing optional runtime settings on
A3, compare the same checkpoint, prompts, seed, and evaluation settings; check
HS contents, loss, gradients, parameter updates, and acceptance statistics. Online
batch scheduling and accelerator kernels can introduce numerical variation, so
bitwise identity across different runtime configurations is not guaranteed.
