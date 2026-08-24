#!/usr/bin/env bash
# RDMA / NCCL environment for DeepSeek-V4-Flash TP=4 across four DGX Sparks.
#
# These are the -e vars the launcher (tp4-4x-spark-launch.sh) passes into the container.
# This file is here as an annotated reference — you can `source` it and export, or just
# read it to understand what each var does. Values are the DGX Spark defaults on a
# SWITCHED RoCE fabric (192.168.192.0/24). Confirm the device names for YOUR rig:
#   ROCE HCA   -> `ibstatus`   (the cabled RoCEv2 HCA)
#   CTRL NIC   -> `ip link`    (the Ethernet control-plane interface)
#   GID index  -> `show_gids`  (the index whose type is "RoCE v2" and address is IPv4)
#
# IMPORTANT: these must be IDENTICAL on all four nodes. In particular NCCL_IB_GID_INDEX=3
# (RoCEv2/IPv4) is used on EVERY node — do not use a different GID on the workers.

# --- The one cabled RoCE HCA + its RoCEv2/IPv4 GID index -----------------------------------
export NCCL_IB_DISABLE=0                 # use IB/RoCE verbs (do NOT disable on the Spark fabric)
export NCCL_IB_HCA=rocep1s0f0            # the ONE HCA on the switched fabric (confirm: ibstatus)
export NCCL_IB_GID_INDEX=3               # RoCEv2 IPv4 GID index — SAME on all four nodes
export NCCL_IB_ROCE_VERSION_NUM=2        # force RoCEv2
export NCCL_IB_ADDR_RANGE=192.168.192.0/24   # the subnet all four Sparks share on the switch

# --- Control-plane (TCP bootstrap) interface ----------------------------------------------
# vLLM's rendezvous + Gloo/TP sockets ride the regular Ethernet NIC, not the RoCE fabric.
CTRL_IF=enp1s0f0np0                      # confirm with `ip link`
export NCCL_SOCKET_IFNAME="$CTRL_IF"
export GLOO_SOCKET_IFNAME="$CTRL_IF"
export TP_SOCKET_IFNAME="$CTRL_IF"

# --- Misc NCCL hygiene --------------------------------------------------------------------
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN                   # bump to INFO to watch which HCA/GID NCCL selects

# --- Per-node vars the LAUNCHER sets (shown here for reference; not exported globally) -----
# Each node also gets, injected by tp4-4x-spark-launch.sh:
#   VLLM_HOST_IP=<this node's own fabric IP>     e.g. 192.168.192.2 on rank 1
#   --node-rank <0|1|2|3>                        rank 0 = head (serves :8000), 1/2/3 = --headless
#   --master-addr <HEAD_FABRIC_IP>               rank 0's fabric IP, same on all nodes
#   --master-port 29640
#
# GB10 arch / cache vars (set in the launcher):
#   TORCH_CUDA_ARCH_LIST=12.1a
#   FLASHINFER_CUDA_ARCH_LIST=12.1a
#   HF_HUB_OFFLINE=1  HF_HOME=/cache/huggingface  VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache
