# Troubleshooting — DeepSeek-V4-Flash TP=4 on four DGX Sparks (GB10)

The real failure modes, and what actually fixed them. Everything the two-Spark recipe hits
also applies here (RDMA init, GID handshake, cold-start, garbage output, node sync, the
FP8-vs-"NVFP4" note) — this file covers those **plus** the failures that only show up once
you go to four nodes on a switched fabric. Ordered roughly by how often people get stuck.

---

## 0. TP=4 needs a SWITCHED fabric — a direct cable won't cut it

**Symptom:** rendezvous hangs forever; some ranks never join; NCCL can't reach a peer.

**Cause:** TP=2 can run over a single **direct** QSFP cable between two Sparks (link-local
`169.254.x.x`). TP=4 needs **all four nodes to reach each other**, which a point-to-point
cable can't do.

**Fix:** put all four Sparks on **one subnet through a RoCE-capable switch** — we use
`192.168.192.0/24` on a MikroTik CRS812. Each node's fabric IP is `.1/.2/.3/.4`; the head's
IP (`.1`) is the `--master-addr` on **all four**. Confirm every node can reach every other
node's fabric IP before launching (`ping 192.168.192.{1,2,3,4}` from each).

---

## 1. Uniform NCCL GID/HCA on ALL FOUR nodes (don't reuse a link-local worker variant)

**Symptom:** the rendezvous hangs, or a QP handshake dies with `ibv_modify_qp ... local
GID ::` (a null GID) on one node.

**Cause:** the DGX Spark NIC exposes multiple GIDs (RoCEv1/v2, IPv4/IPv6) and two RoCE
ports. On the **switched** fabric the RoCEv2/IPv4 entry is **GID index 3 on every node**.
Older two-node, direct-cable scripts sometimes used a different GID on the worker — carry
that over and the worker builds a queue pair on the wrong/empty GID and the handshake dies.

**Fix:** pin **all four** nodes to the single cabled HCA and GID index 3:

```bash
-e NCCL_IB_HCA=rocep1s0f0
-e NCCL_IB_GID_INDEX=3
-e NCCL_IB_ROCE_VERSION_NUM=2
```

Verify with `show_gids` on **each** node: pick the index whose type is `RoCE v2` and whose
address is that node's IPv4 on the fabric. It should be the same index (3) on all four. To
debug, set `-e NCCL_DEBUG=INFO` and read which HCA/GID each rank selects.

---

## 2. RDMA won't initialize — NCCL "unhandled system error" / "Cannot allocate memory"

**Symptom:** cluster init dies during `ibv_reg_mr` on a node.

**Cause:** that container can't pin (lock) memory or can't see the IB verbs devices.

**Fix:** every node's container **must** run with all of these:

```bash
--privileged \
--ulimit memlock=-1 \
--device /dev/infiniband:/dev/infiniband \
--network host --ipc host --shm-size 10g
```

`--privileged` + `--ulimit memlock=-1` is the pair that lets RDMA pin its memory regions.
Mapping `/dev/infiniband` exposes the verbs devices. Host networking is required so the
container sees the real fabric. Confirm RDMA on the host first (`ibstatus` ACTIVE,
`show_gids` valid RoCEv2 GID) — on **all four** nodes.

---

## 3. Single-quote the JSON args, or all four nodes exit(2) instantly

**Symptom:** every node's container exits immediately (before any weight load) with an
argparse error like `invalid loads value: {method:mtp,num_speculative_tokens:2}`.

**Cause:** the serve command runs inside the container via `bash -lc "..."`. If the JSON
args (`--speculative-config`, `--default-chat-template-kwargs`) are **double-quoted**, the
inner shell strips the double-quotes and argparse gets malformed JSON.

**Fix:** keep JSON args **single-quoted** inside the inner command:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

This is a **cheap, fast** failure (it dies before loading anything), which is exactly why we
recommend a clean first boot before adding anything fancy — you diagnose it in seconds.

---

## 4. Startup order — THREE workers first, then the head

**Symptom:** the head comes up, starts the API/rendezvous, and NCCL init stalls because
peers aren't listening yet.

**Cause:** at TP=4 there are **three** workers, not one. If the head (rank 0) starts before
ranks 1/2/3 are up, the multi-node rendezvous can hang.

**Fix:** launch **ranks 3, 2, 1 first** (all `--headless`), wait ~20s, then the head (rank
0). Watch the head with `docker logs -f vllm_ds4_tp4` and wait for `Application startup
complete`. Because each node loads only ¼ the weights this is fast (~5 min).

---

## 5. Cold first request times out (but the server is fine)

**Symptom:** the very first completion after launch (or a long idle) times out on the
client, even though `GET /v1/models` returns 200 and a retry works instantly.

**Cause:** cold compilation / MoE-expert warmup for the new prompt shape (torch.compile +
cudagraph capture + kernel JIT). The server didn't fail — the client's timeout fired first.

**Fix:** just retry — the warm path is instant. Real traffic keeps it warm. If you must
avoid it, raise the client's first-call timeout or fire a tiny `max_tokens=3` warm-up.

**Tell it apart from a real failure:** if `/v1/models` is 200 and a *second* completion
works, it was cold-start — don't chase RDMA. If `/v1/models` is down or completions hang
*repeatedly*, that's a real problem (wedge or fabric).

---

## 6. Fresh `docker run` only — `drop_caches` first, no `docker restart`, no `--rm`

**Symptom:** a relaunch fails with a driver-OOM at boot, or a crash leaves you with no
container and no logs, or `docker restart` brings up a wedged cluster.

**Fixes (all three):**

- **`drop_caches` on all four before launch:** `sync; echo 3 > /proc/sys/vm/drop_caches`.
  The driver can hold ~100 GiB of unified memory; a stale cache causes a driver-OOM boot
  fail.
- **Fresh `docker run`, not `docker restart`** — this cluster does not survive a restart.
  Tear down first on all four: `docker rm -f vllm_ds4_tp4`.
- **Don't use `--rm`.** On a crash it auto-removes the container and takes the logs with it.
  Use plain `docker run -d` (logs persist) or `--restart=unless-stopped` for auto-recovery.
  A multi-node cluster still needs *all four* nodes back for service to resume.

---

## 7. Node sync matters more than any flag — and there are FOUR nodes now

The single most important thing. **The biggest performance unlock is not a vLLM flag — it's
making every node bit-for-bit identical** on NVIDIA **driver**, **kernel**, and **SoC
firmware** (EC + SoC + USB-C PD capsules). A mismatch silently bottlenecks a node on the
`sm_121a` codepath; on our two-Spark rig, syncing was worth **~+140% prefill** — more than
any single flag. At four nodes there are **four chances** to be out of sync, so check all
four:

```bash
nvidia-smi        # driver version
uname -r          # kernel
# + your firmware/EC version check
```

Make them match, and reboot after any driver/firmware change before the first run.

---

## 8. "NVFP4" is a mirage — the weights are actually FP8

Lots of community material labels the GB10 DS4 builds "NVFP4." **That labeling is wrong.**
Every result we and others have measured is native **FP8** (E4M3 block-scaled dense + mxfp4
MoE). Practical consequences:

- Don't hunt for an "NVFP4" speed advantage — it isn't there; you're running FP8.
- A healthy FP8 fast-path shows up in startup logs as something like
  `Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM` — that's the
  native FP8 DeepGEMM path. If you *don't* see it, you can fall to ~5 t/s (roughly 7×
  slower) — the trap that makes people think the model is "broken."

---

## 9. Garbage / incoherent output after an image or build change

**Symptom:** the model loads and serves but emits digit-soup / garbled tokens.

**Fixes:**

- **Stale compile caches.** Wipe the vLLM / Triton / TileLang / DeepGEMM caches between
  image or build changes (they live under the mounted HF cache). Stale JIT artifacts
  silently garble output.
- **Wrong arch kernels.** Confirm you're on the `sm_121` / `12.1a` build
  (`TORCH_CUDA_ARCH_LIST=12.1a`). A mismatched-arch kernel can produce garbage.
- **Smoke-test after every change** — send one short prompt and confirm real text comes back
  before declaring the deploy healthy.

---

## Quick triage flowchart

```
Service down?
├─ some ranks never join / rendezvous hangs .... switched fabric or GID mismatch -> #0, #1
├─ GET /v1/models fails ........................ cluster not up / RDMA failing -> #1, #2
├─ all four containers exit instantly .......... JSON args double-quoted -> #3
├─ /v1/models 200 but FIRST completion timed out  cold-start -> #5 (just retry)
├─ /v1/models 200 but completions hang REPEATEDLY decode wedge -> fresh run all four (#6)
└─ Output is garbage ........................... stale caches / wrong arch -> #9

Slow?
├─ ~5 t/s ...................................... native FP8 path not engaged -> #8
├─ one node bottlenecks the group ............. node version mismatch -> #7 (sync all four)
└─ want more concurrency vs more ctx .......... tune --max-num-seqs / --max-model-len (README)
```
