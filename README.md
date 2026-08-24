# DeepSeek-V4-Flash On FOUR NVIDIA DGX Spark (GB10): TP=4 Recipe

> **Unofficial community recipe.** Not affiliated with, endorsed by, or supported by
> DeepSeek, NVIDIA, or the vLLM project. All names and trademarks belong to their owners.
> Every number labeled "our measurement" is from our own rig — treat it as *relative*, not
> gospel, and benchmark your own prompt mix.

A complete, working recipe for serving **DeepSeek-V4-Flash** across **four NVIDIA DGX Spark
(GB10) boxes** as **one tensor-parallel group (TP=4)** with vLLM, over a switched 200GbE
RoCE/RDMA fabric.

This is the **four-Spark sibling** of our two-Spark recipe
([`tonyd2wild/Deepseek-v4-Flash-TP2-DGX-Spark-500k-CTX`](https://github.com/tonyd2wild/Deepseek-v4-Flash-TP2-DGX-Spark-500k-CTX)).
The RDMA/NCCL setup, container flags, worker-first startup, and `--distributed-executor-backend mp`
handshake are **identical** to the TP=2 recipe — the only new requirement is a **switched
fabric** so all four nodes can reach each other. Read the TP=2 repo first if you're new to
DS4-on-Spark; this one focuses on what changes at four nodes.

---

## TL;DR — the surprise

On these boxes, **TP=4 is *faster* on single-stream decode than TP=2**, not slower — and it
gives you a **~4x larger KV pool** and enough concurrency to serve several agent sessions at
once from **one** endpoint (no relay). That is the opposite of what happens on
PCIe-connected multi-GPU boxes. The Spark's fast RoCE fabric flips the result. It surprised
us; the benchmarks below are why we now consider TP=4 DS4 a real contender for burning all
four Sparks on one model.

---

## The benchmarks (our measurement) — C1–C6 across three prompt classes

**Method:** non-streaming, `temperature=0`, `max_tokens=200`, a **unique nonce at the front
of every request** (so vLLM's prefix cache can't inflate the numbers), N concurrent
identical-class requests per level (C1 = 1 stream … C6 = 6 streams). `per-stream` =
mean(`completion_tokens / wall`) across the N requests; `aggregate` = total tokens ÷ batch
wall-clock. Three prompt classes:

- **count** — highly predictable output → high speculative-accept → **best case**
- **code** — structured, medium accept
- **prose** — freeform, low accept → the realistic **worst case**

### C1 (single stream): TP=4 beats TP=2 in every class

| class | TP=2 C1 | **TP=4 C1** | Δ |
|-------|--------:|------------:|:-:|
| count | 87 t/s  | **120 t/s** | **+37%** |
| code  | 66 t/s  | **87 t/s**  | **+32%** |
| prose | 42 t/s  | **46 t/s**  | **+9%**  |

Prose gains least because freeform text accepts the fewest draft tokens; count gains most.
Even so, **TP=4 is ahead on all three.** (Our count-to-300 headline stream peaked at ~127
t/s; the 120 above is the strict nonce-front, `max_tokens=200` count class.)

### C1→C6: TP=4 aggregate throughput as you add concurrent streams (total tokens/sec)

| C (streams) | count | code | prose |
|:-----------:|------:|-----:|------:|
| **C1** | 120 | 87  | 46  |
| **C2** | 181 | 123 | 70  |
| **C3** | 239 | 142 | 84  |
| **C4** | 293 | 188 | 106 |
| **C5** | 300 | 171 | 107 |
| **C6** | **402** | **251** | **135** |

Per-stream at C6: **count ~69, code ~45, prose ~24 t/s each.** Aggregate scales **~3x** from
C1→C6. The occasional C5 dip (e.g. code 188→171) is scheduler-batching noise — C6 recovers.

**The takeaway most people get wrong:** the common objection to TP=4 is *"concurrency will
suck because it's one instance instead of two lanes."* It doesn't. A single TP=4 instance
batches **6 concurrent streams** and still gives every user usable speed — **more aggregate
throughput than two TP=2 lanes doing ~80/80**, **plus** ~4x the context headroom, **plus**
one endpoint instead of a relay in front of two lanes.

---

## TP=2 vs TP=4, side by side (our measurement)

| | TP=2 (2 Sparks) | **TP=4 (4 Sparks)** |
|---|---|---|
| Weights held **per node** | ~80 GB (½ the model) | **~38 GB (¼ the model)** |
| GPU KV cache pool | ~1.3–1.5M tokens | **~6M+ tokens (~4x)** |
| Concurrency headroom @ ~888K/req | ~1.6x | **~7.6x** |
| Single-stream decode, count | ~87 t/s | **~120 t/s (peak ~127)** |
| 2 concurrent streams (count) | ~80/80 across two lanes | **100/100 on one instance** |
| Max context, single request | up to ~500K | **up to ~1M** (the pool supports it) |
| Boot time | ~12 min | **~5 min** (each node loads only ¼ the weights) |
| Ops shape | two lanes + a relay in front | **one instance, one endpoint** |

### Why TP=4 is *faster*, not slower (the counter-intuitive part)

DS4 single-stream decode on these boxes is **memory-bandwidth bound**, not compute bound. At
TP=4 each Spark holds only **~¼ of the weights** (~38 GB vs ~80 GB at TP=2), so it reads a
quarter of the weight memory per generated token → **decode speeds up**. The cost that would
normally cancel that out — a **4-way** all-reduce every layer instead of 2-way — rides the
**200GbE RoCE fabric**, which is fast enough that the per-node bandwidth win dominates.

This is the **opposite** of PCIe-connected multi-GPU boxes (e.g. 4× 3090), where TP=4
all-reduces crawl over PCIe and TP=4 comes out *slower*. **Do not carry that "TP=4 is
slower" intuition to the Sparks** — the fast fabric flips it. Measure it on your own rig.

---

## Topology

```
                 switched RoCE fabric — 192.168.192.0/24  (200GbE, RoCEv2, MTU 9000)
       ┌──────────────┬──────────────┬──────────────┬──────────────┐
       │              │              │              │
  ┌────┴─────┐   ┌────┴─────┐   ┌────┴─────┐   ┌────┴─────┐
  │  Spark 0 │   │  Spark 1 │   │  Spark 2 │   │  Spark 3 │
  │  rank 0  │   │  rank 1  │   │  rank 2  │   │  rank 3  │
  │  HEAD    │   │  worker  │   │  worker  │   │  worker  │
  │ .1  :8000│   │  .2      │   │  .3      │   │  .4      │
  │ ¼ weights│   │ ¼ weights│   │ ¼ weights│   │ ¼ weights│
  └────┬─────┘   └──────────┘   └──────────┘   └──────────┘
       │  serves API      --headless    --headless    --headless
       ▼
  OpenAI-compatible  http://<HEAD_IP>:8000/v1   ← clients / agents
```

- **`--tensor-parallel-size 4 --nnodes 4`**, ranks 0–3, one GPU per Spark.
- **`--distributed-executor-backend mp`** (multiprocessing) — **not** Ray. The proven
  multi-node path is `--nnodes / --node-rank / --master-addr` directly with `mp`.
- **Master** = rank 0's fabric IP (`192.168.192.1` on our rig), rendezvous port `29640`.
- **Uniform NCCL env on all four nodes:** `NCCL_IB_HCA=rocep1s0f0`, `NCCL_IB_GID_INDEX=3`
  (RoCEv2/IPv4 — same index on **every** node), control-plane socket interface, MTU 9000.

---

## Hardware requirements

| Component | Requirement |
|---|---|
| **Compute** | **4× NVIDIA DGX Spark** (GB10 Grace-Blackwell, `sm_121` / arch `12.1a`). One GPU each; vLLM runs TP=4 across the four. |
| **Interconnect** | A **switched** 200GbE RoCE/RDMA fabric — every Spark on one subnet (we use `192.168.192.0/24`) through a RoCE-capable switch (we use a MikroTik CRS812). **This is the one real difference from TP=2**, which can run over a single direct QSFP cable. TP=4 needs all four nodes to reach each other. |
| **Unified memory** | GB10 is **unified CPU+GPU memory (~120 GB/node)**. At TP=4 the model is only ~38 GB/node, so you have a *lot* of headroom for KV pool and co-tenancy. |
| **Disk** | Model weights (~150 GB) + Docker image (~36 GB) on **each** node. Mount the host HF cache into the container. |
| **Software (matters most)** | **Matched driver + kernel + SoC firmware across ALL FOUR nodes.** A version mismatch silently bottlenecks a node on the `sm_121a` codepath. Node sync was worth more than any single vLLM flag in our TP=2 testing (+140% prefill) — and at four nodes there are four chances to be out of sync. |
| Container runtime | Docker + NVIDIA Container Toolkit (`--gpus all`), `--privileged`, host networking, `--device /dev/infiniband`. |

Confirm **RDMA works at the OS level first** on every node — `ibstatus` (HCA `ACTIVE`) and
`show_gids` (a valid RoCEv2 GID). If the host can't do RDMA, the container never will.

---

## The head-count rule — why TP=4 and *not* TP=3

DeepSeek-V4-Flash has **128 attention heads**. The TP size must divide the head count:

- **TP=4 works** → 128 ÷ 4 = 32 ✅
- **TP=3 does not** → 128 ÷ 3 is not an integer ❌ (for three nodes you'd fall back to
  pipeline-parallel, which is a different, slower path)

Four Sparks → TP=4 is the clean tensor-parallel path. Don't try TP=3.

---

## The Docker image

```
aidendle94/sparkrun-vllm-ds4-gb10:production-ready
```

- Public on Docker Hub, `arm64` (the Spark is ARM/Grace), ~36 GB unpacked.
- vLLM built native for `sm_121` (GB10) with the DS4-specific kernels + an entrypoint
  (`/usr/local/bin/dsv4-vllm-entrypoint`) that wraps `vllm serve`.
- **Pre-pull on ALL FOUR nodes** before launching (a cold pull mid-deploy stalls the
  handshake):

  ```bash
  docker pull aidendle94/sparkrun-vllm-ds4-gb10:production-ready
  ```

> The launcher in this repo uses this public image and the model's native **MTP**
> speculative decoding (`num_speculative_tokens=2`) so anyone can reproduce it. In our own
> production deployment we run the same **DSpark spec-decode "winner" stack** we use on the
> TP=2 lane (`k=5` dspark drafter + fused-Markov-argmax + the drafter-sizes patch); DSpark
> spec-decode works fine at TP=4 (proven — decode stays clean, no garble). That stack rides
> a private image and an internal model path, so it's not published here; the public MTP
> config below is the reproducible recipe and is what every number in the tables was
> **not** measured on unless noted — see the note under the benchmark method.

---

## Quick start (TL;DR)

1. Pre-pull the image and pre-download the weights into each node's HF cache — on **all
   four** nodes.
2. Put all four Sparks on the **switched RoCE fabric** (`192.168.192.0/24`), matched
   driver/kernel/firmware, RDMA up (`ibstatus`, `show_gids`).
3. Edit [`launch/tp4-4x-spark-launch.sh`](launch/tp4-4x-spark-launch.sh): set `MASTER_ADDR`
   to the head's fabric IP and confirm the RoCE HCA / control NIC / GID index for your rig.
4. **Start the three WORKERS first (ranks 3, 2, 1), then the HEAD (rank 0)** — each node
   passes its own fabric IP:

   ```bash
   # on each worker (its own fabric IP as arg 2):
   ./launch/tp4-4x-spark-launch.sh 3 192.168.192.4
   ./launch/tp4-4x-spark-launch.sh 2 192.168.192.3
   ./launch/tp4-4x-spark-launch.sh 1 192.168.192.2
   # wait ~20s, then the HEAD:
   ./launch/tp4-4x-spark-launch.sh 0 192.168.192.1
   ```
5. Watch the head come up (`docker logs -f vllm_ds4_tp4` on rank 0). Because each node loads
   only ¼ the weights this is **fast — ~5 min**, vs ~12 for a node holding half at TP=2.
   Wait for `Application startup complete`.
6. The **head** serves an OpenAI-compatible API on `:8000`:

   ```bash
   curl http://<HEAD_IP>:8000/v1/models
   ```

   **First request after a cold start may time out** — retry; the warm path is instant (see
   [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)).

Point any OpenAI-compatible client at `http://<HEAD_IP>:8000/v1` — see
[`AGENT-WIRING.md`](AGENT-WIRING.md).

---

## The serve command (what the launcher runs)

```bash
dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash-spark deepseek-v4-flash \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 300000 --max-num-seqs 6 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.78 --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --enable-flashinfer-autotune \
  --nnodes 4 --node-rank <0|1|2|3> --master-addr <HEAD_FABRIC_IP> --master-port 29640 \
  <--headless on workers only>
```

### Every flag that differs from (or matters more than) the TP=2 recipe

| Flag | Value | Why (at TP=4) |
|---|---|---|
| `--tensor-parallel-size 4` | `4` | All four Sparks in one TP group; each holds ¼ the model. **Must divide 128 heads** → 4 ✅, 3 ❌. |
| `--nnodes 4` | `4` | Four nodes in the cluster. |
| `--node-rank <0..3>` | 0 head / 1,2,3 workers | This node's rank. Rank 0 serves the API; the rest are `--headless`. |
| `--master-addr <HEAD_FABRIC_IP>` | head's **fabric** IP | Rendezvous address — the head's IP on the **switched RoCE fabric** (e.g. `192.168.192.1`), the same on all four nodes. |
| `--master-port 29640` | `29640` | Rendezvous port. |
| `--distributed-executor-backend mp` | `mp` | Multiprocessing, **not Ray**. Proven multi-node path with `--nnodes/--node-rank`. |
| `--gpu-memory-utilization 0.78` | `0.78` | Because weights are only ¼/node, the KV pool this carves out is **~6M tokens**. The gmu *floor* is much lower than TP=2's ~0.72 — you have real headroom; you can even raise it or co-tenant another workload. |
| `--max-model-len 300000` | `300000` | Keeps per-request KV small so concurrency stays high (great for agents). Raise toward **1000000** for giant single-context runs — the ~6M-token pool supports it; you trade some concurrency for context. |
| `--max-num-seqs 6` | `6` | Comfortable and well inside the KV headroom — **you can push it higher**. (The TP=2 recipe uses 8; at TP=4 you have *more* room, not less. 6 is a conservative default that maps to the C1–C6 sweep.) |
| `--kv-cache-dtype fp8` | `fp8` | FP8 KV — roughly halves KV memory vs fp16. Essential on the unified-memory budget. |
| `--block-size 256` | `256` | Larger KV block; pairs well with FP8 KV and the DS4 sparse-MLA path. |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` | MTP, 2 | Native Multi-Token Prediction drafter. Speed is **class-dependent** — predictable output (count/structured) accepts nearly every draft token and flies; freeform prose accepts less. That's the whole story behind the C1 count-vs-prose gap. |
| `--tokenizer-mode deepseek_v4` | — | DS4 tokenizer mode. |
| `--tool-call-parser deepseek_v4` `--enable-auto-tool-choice` `--reasoning-parser deepseek_v4` | — | Native tool/function-calling + reasoning-field parsing. |
| `--served-model-name ...` | aliases | Serves the model under multiple names at once so you can swap the underlying model later without re-pointing clients. |

> **Editing the launcher?** The JSON args (`--speculative-config`, etc.) are **single-quoted**
> so they survive the container's inner `bash -lc`. If you double-quote them the inner shell
> strips the quotes → `{method:mtp}` → argparse `invalid loads value` and **all four nodes
> exit(2) instantly**. Keep them single-quoted. (This cost us a boot.)

---

## Tuning notes — headroom & KV

At TP=4 the constraint that dominates TP=2 (fitting half the model *plus* a usable KV pool
into one node's unified memory) largely goes away:

- **~38 GB weights/node** leaves **~30 GB+ free per node** beyond vLLM's 0.78 allocation.
  You can push `--max-model-len` far past 300K, raise `--gpu-memory-utilization`, raise
  `--max-num-seqs`, **or** co-tenant another workload (e.g. an image model) on every node.
- **KV pool ≈ 6M+ tokens** at gmu 0.78 (vs ~1.3–1.5M at TP=2). That's what powers the
  ~7.6x concurrency headroom at ~888K/request and the up-to-~1M single-request context.
- **max-num-seqs**: 6 maps to the C1–C6 sweep, but the pool supports many more concurrent
  streams at moderate context. Raise it for a bigger agent fleet.
- **Speculative decoding is class-dependent** — benchmark your *own* prompt mix. Counting
  and structured/JSON output fly; freeform prose is the floor. Don't tune on count alone.

---

## Gotchas specific to TP=4 (read before you launch)

1. **You need a switched fabric.** TP=2 can run over one direct QSFP cable (link-local
   `169.254.x.x`). TP=4 needs all four nodes routable to each other — every Spark on one
   subnet through a RoCE switch. This is the single real config difference from TP=2.
2. **Uniform NCCL GID on all four nodes.** Use `NCCL_IB_GID_INDEX=3` (RoCEv2/IPv4) and
   `NCCL_IB_HCA=rocep1s0f0` **on every node**. Do **not** carry over any "GID-5-for-worker"
   variant from older two-node link-local scripts — the switched fabric wants GID 3
   everywhere. A mismatched GID/HCA on any one node hangs the whole rendezvous.
3. **Worker-first, and it's *three* workers now.** Bring up ranks **3, 2, 1** (all
   `--headless`), wait ~20s, then the head (rank 0). If the head starts before the workers
   are listening, the NCCL init can stall.
4. **Single-quote the JSON args** (see the boxed note above) — double-quoting exits all four
   nodes instantly before any load.
5. **Four nodes = four chances to be out of sync.** Match driver/kernel/firmware on **all
   four**; one odd node silently bottlenecks the group on the `sm_121a` path.
6. **`drop_caches` on all four before a fresh launch** (`sync; echo 3 > /proc/sys/vm/drop_caches`)
   to avoid a driver-OOM boot fail, and always **fresh `docker run`** — this cluster does not
   survive `docker restart`. Tear down any prior container on all four first
   (`docker rm -f vllm_ds4_tp4`).

More failure modes (RDMA init, GID `local GID ::`, cold-start, garbage output, the `--rm`
pitfall, node sync, and the FP8-vs-"NVFP4" note) are in
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — they're shared with the TP=2 recipe.

---

## Files in this repo

- [`launch/tp4-4x-spark-launch.sh`](launch/tp4-4x-spark-launch.sh) — the scrubbed TP=4
  launcher (run the same script on all four nodes; args = rank + that node's fabric IP).
- [`launch/nccl-env.example.sh`](launch/nccl-env.example.sh) — the RDMA/NCCL env block,
  annotated, if you'd rather export it than inline it.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — the real failure modes and what fixed them.
- [`AGENT-WIRING.md`](AGENT-WIRING.md) — point any OpenAI-compatible client/agent at the
  head's `:8000/v1`.

## See also

- **TP=2 (two-Spark) recipe:**
  [`tonyd2wild/Deepseek-v4-Flash-TP2-DGX-Spark-500k-CTX`](https://github.com/tonyd2wild/Deepseek-v4-Flash-TP2-DGX-Spark-500k-CTX)
  — start there for the base DS4-on-Spark setup, the RDMA/NCCL deep-dive, and agent wiring.
- The **"NVFP4" is a mirage** note (the weights are FP8) and the **node-sync** finding both
  carry over verbatim — see this repo's `TROUBLESHOOTING.md`.

---

*MIT licensed. Unofficial — a community recipe, not an official DeepSeek / NVIDIA / vLLM
artifact.*
