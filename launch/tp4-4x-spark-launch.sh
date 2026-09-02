#!/usr/bin/env bash
# DeepSeek-V4-Flash on FOUR NVIDIA DGX Spark (GB10) — vLLM multi-node, TENSOR-PARALLEL 4.
#
# This is the 4-node ("TP4") sibling of our two-Spark TP=2 recipe. Instead of two Sparks at
# TP=2, all FOUR Sparks join ONE tensor-parallel group. Each node then holds only a QUARTER of
# the weights, which (counter-intuitively) makes single-stream decode FASTER, not slower,
# on the Spark's fast RoCE fabric — see README.md for the why and the numbers.
#
# WHY TP4 IS DIFFERENT FROM TP2 ON THE FABRIC:
#   TP=2 can run over a single QSFP cable directly between two Sparks (link-local 169.254.x).
#   TP=4 needs all four nodes to reach each other, so you need a SWITCHED RoCE fabric —
#   every Spark on one subnet (e.g. 192.168.192.0/24) through a RoCE-capable switch. The
#   all-reduce is 4-way instead of 2-way; on the 200GbE fabric it's cheap enough that the
#   per-node memory-bandwidth win dominates.
#
# HEAD-COUNT RULE: DeepSeek-V4-Flash has 128 attention heads. TP size must divide 128, so
#   TP=4 (128/4=32) is valid; TP=3 is NOT (use TP=2 or TP=4, not 3 — for 3 nodes you'd need
#   pipeline-parallel instead).
#
# Run this SAME script on ALL FOUR nodes; args are the node rank and that node's fabric IP:
#   arg1 = NODE_RANK   0 = head (serves the API on :8000) | 1,2,3 = workers (--headless)
#   arg2 = this node's own IP on the switched RoCE fabric (e.g. 192.168.192.2)
#
# STARTUP ORDER: start the THREE WORKERS (ranks 3, 2, 1) FIRST, then the HEAD (rank 0).
#
# Fill in the <PLACEHOLDERS> for your rig (see README.md for the RDMA/NCCL notes —
# they are identical here; the only new requirement is the switched fabric + per-node IP):
#   <HEAD_FABRIC_IP>  - the HEAD node's IP on the switched RoCE fabric (rendezvous master).
#   rocep1s0f0        - your cabled RoCE HCA device name (confirm with `ibstatus`).
#   enp1s0f0np0       - your control-plane Ethernet interface (confirm with `ip link`).
#   NCCL_IB_GID_INDEX - the RoCEv2/IPv4 GID index on that HCA (confirm with `show_gids`).

set -uo pipefail
NODE_RANK="${1:?usage: tp4-4x-spark-launch.sh <0|1|2|3> <THIS_NODE_FABRIC_IP>}"
HOST_IP="${2:?usage: tp4-4x-spark-launch.sh <0|1|2|3> <THIS_NODE_FABRIC_IP>}"
HEADLESS_FLAG=""
[ "$NODE_RANK" != "0" ] && HEADLESS_FLAG="--headless"

MASTER_ADDR="<HEAD_FABRIC_IP>"     # rank 0's IP on the switched RoCE fabric
MASTER_PORT="29640"
FABRIC_SUBNET="192.168.192.0/24"   # the subnet all four Sparks share on the switch
CTRL_IF="enp1s0f0np0"              # TCP control-plane NIC
ROCE_HCA="rocep1s0f0"              # the RoCE HCA on the switched fabric

docker rm -f vllm_ds4_tp4 2>/dev/null || true

docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  --name vllm_ds4_tp4 \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$ROCE_HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ADDR_RANGE="$FABRIC_SUBNET" -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_SOCKET_IFNAME="$CTRL_IF" -e GLOO_SOCKET_IFNAME="$CTRL_IF" -e TP_SOCKET_IFNAME="$CTRL_IF" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  --entrypoint bash \
  aidendle94/sparkrun-vllm-ds4-gb10:production-ready \
  -lc "exec /usr/local/bin/dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name deepseek-v4-flash-spark deepseek-v4-flash \
    --host 0.0.0.0 --port 8000 --trust-remote-code \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 300000 --max-num-seqs 6 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.78 --enable-prefix-caching \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}' \
    --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
    --enable-flashinfer-autotune \
    --nnodes 4 --node-rank $NODE_RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT $HEADLESS_FLAG"

echo "launched vllm_ds4_tp4 node-rank=$NODE_RANK host=$HOST_IP headless='$HEADLESS_FLAG' rc=$?"
sleep 2
docker ps --format '{{.Names}} | {{.Status}}' | grep vllm_ds4_tp4 || echo "WARN: container not in ps (check: docker logs vllm_ds4_tp4)"

# ---------------------------------------------------------------------------
# COPY-PASTE RUN COMMANDS  (worker-first: ranks 3, 2, 1, then head 0)
#
#   # on each WORKER, passing that node's own fabric IP:
#   ./tp4-4x-spark-launch.sh 3 192.168.192.4
#   ./tp4-4x-spark-launch.sh 2 192.168.192.3
#   ./tp4-4x-spark-launch.sh 1 192.168.192.2
#   # wait ~20s, then the HEAD:
#   ./tp4-4x-spark-launch.sh 0 192.168.192.1
#
#   # watch the head come up (each node loads only 1/4 of the weights, so this is FAST,
#   # ~5 min vs ~12 for a single node holding half at TP2):
#   docker logs -f vllm_ds4_tp4            # on the head node
#   # wait for: "Application startup complete"
#
#   curl http://<HEAD_NODE_IP>:8000/v1/models
#
# NOTES
# - max-num-seqs 6 is comfortable here; the KV pool is ~4x larger than TP=2, so the
#   scheduler has plenty of headroom (you can push seqs higher). See README.md.
# - max-model-len 300000 keeps concurrency high for agents; the ~6M-token KV pool can
#   serve up to ~1M-token single requests if you raise it.
# - JSON args (--speculative-config, etc.) are single-quoted so they survive the inner
#   `bash -lc`; keep them that way if you edit this script.
# ---------------------------------------------------------------------------
