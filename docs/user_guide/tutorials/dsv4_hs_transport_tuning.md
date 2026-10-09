# DSV4 hidden-state transport memory tuning

These controls change host-side staging and waiting, not the exported tensor
dtype, target layer selection, training objective, or acceptance calculation.
Measure the generation, download and load stages before enabling an optimization.
CPU tests cover the ownership and failure cases; A3/NPU and RDMA throughput must
still be measured on the deployment hardware.

## HTTP snapshots

The sidecar keeps checksum and response bytes tied to one immutable snapshot.
Small snapshots can stay in memory. Larger snapshots, or requests that cannot
reserve memory, use temporary files without waiting for another request.

| Sidecar environment variable | Default | Meaning |
| --- | --- | --- |
| `HS_HTTP_SNAPSHOT_MEMORY_BYTES` | `8388608` (8 MiB) | Maximum payload size of one in-memory snapshot |
| `HS_HTTP_SNAPSHOT_TOTAL_MEMORY_BYTES` | `268435456` (256 MiB) | Combined payload budget of all active in-memory snapshots |
| `HS_HTTP_SNAPSHOT_TEMP_DIR` | unset | Existing local directory for spilled snapshots; otherwise Python's temporary directory |

These map to `--snapshot-memory-bytes`, `--snapshot-total-memory-bytes` and
`--snapshot-temp-dir` on `python -m speculators_dsv4.hs_http_server`. Either memory
limit may be zero to force disk snapshots. Limits exclude per-request transfer
chunks, Python objects and operating-system caches, so they are not an RSS cap.
Each reservation stays active until the response closes its snapshot. A growing
source spills before exceeding the reservation and is still rejected by the
existing file-version and checksum checks.

For example, after creating a local spill directory, allow up to four 256 MiB
snapshot reservations:

```bash
HS_HTTP_SNAPSHOT_MEMORY_BYTES=268435456 \
HS_HTTP_SNAPSHOT_TOTAL_MEMORY_BYTES=1073741824 \
HS_HTTP_SNAPSHOT_TEMP_DIR=/local/hs-spill \
bash examples/train/dspark_dsv4_hs_http_server.sh
```

The default 8 MiB threshold is unchanged. A 3072-token sample with six 4096-wide
BF16 states contains about 144 MiB of HS payload before file overhead. Such a
sample goes directly to disk at the default threshold, avoiding an initial
in-memory copy before spill. Do not put the spill directory on the slow shared
mount that HTTP was intended to avoid. Failed downloads and producer files retain
the existing cleanup and retry rules.

## Optional Mooncake producer staging

Both optimizations are disabled by default and are independent of TCP versus
RDMA. The server wrapper accepts these explicit overrides:

```bash
HS_BACKEND=mooncake \
MOONCAKE_NAMESPACE=run-20261009-a \
MOONCAKE_PINNED_POOL_MIB=256 \
MOONCAKE_COPY_EVENTS=1 \
bash examples/train/dspark_dsv4_flash_bf16_server.sh
```

Keep the existing matching namespace, master, topology and trainer settings.
The corresponding launcher options are `--mooncake-pinned-pool-mib 256` and
`--mooncake-copy-events`, before the `--` separator. With no overrides, the reuse
budget is zero and the producer retains its previous stream-wait behavior.

The pool reuses exact shape/dtype host buffers only after the synchronous store
write finishes. Its budget includes both idle and borrowed pooled buffers, per
producer process. A busy/full pool uses an unpooled allocation rather than
blocking the writer; therefore this budget does not cap all in-flight host
memory. Devices without pinned-memory support keep the existing ordinary-CPU
allocation fallback. Use the writer count and host-memory measurements alongside
the pool limit.

Copy events wait for the individual DtoH completion before publishing to the
store. A failed copy/event first drains the copy stream. If completion cannot be
established, its buffer is retained and the connector raises instead of reporting
the request's KV as safe to release. Failed store writes do not return the buffer
to the reuse pool.

Compare fixed input batches with options off/on: tensor dtype, shape, token IDs,
checksums and values must match. Also measure host RSS, pinned allocation counts,
HS throughput and p95 latency under concurrent requests. A faster copy stage is
not proof of an end-to-end training speedup.
