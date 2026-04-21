#!/bin/bash
#
# Example SBATCH script for serving a model with vLLM on the HPC cluster.
# Copy and adapt this file for each model you want to run.
#
# Usage:
#   sbatch example_sbatch.sh
#
# Logs are written to <job-name>-<job-id>.out / .err

# ── SLURM directives ───────────────────────────────────────────────────────────
#SBATCH --job-name=vllm-llama-3.1-8b
#SBATCH --partition=gpu-8-h100       # Change to the partition with your target GPU
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=2-00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ── Environment ────────────────────────────────────────────────────────────────
module load singularity
module load compilers/nvidia/cuda/12.6
module load softwares/squashfs

# Path to the Singularity image (built from vllm.def)
SIF=/path/to/your/vllm.sif

# Model identifier on Hugging Face Hub
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

# Port exposed by the vLLM server (use a different port per model)
PORT=8000

# Your Hugging Face token (for gated models such as Llama)
# Set via environment variable instead of hardcoding:
#   export HF_TOKEN=hf_...
HF_TOKEN=${HF_TOKEN:?"HF_TOKEN is not set. Export it before submitting the job."}

# ── Cache directories ──────────────────────────────────────────────────────────
HF_CACHE_DIR="/home/$USER/hf_cache"
mkdir -p "$HF_CACHE_DIR/transformers" "$HF_CACHE_DIR/hub"

# ── pyairports stub (vLLM dependency workaround) ───────────────────────────────
# vLLM imports pyairports transitively; this stub prevents import errors
# without modifying the container.
PYAIRPORTS_DIR="/home/$USER/pyairports_stub"
mkdir -p "$PYAIRPORTS_DIR/pyairports"
echo "AIRPORT_LIST = []"  > "$PYAIRPORTS_DIR/pyairports/airports.py"
echo ""                   > "$PYAIRPORTS_DIR/pyairports/__init__.py"

# ── vLLM server command ────────────────────────────────────────────────────────
CMD="vllm serve $MODEL
    --tensor-parallel-size 1
    --host 0.0.0.0
    --port $PORT
    --max-model-len 16384
    --dtype half
    --gpu-memory-utilization 0.90
    --max-num-seqs 16
    --max-num-batched-tokens 16384
    --disable-custom-all-reduce"

# Add --enforce-eager for models that fail without it (e.g. some Mistral versions)
# CMD="$CMD --enforce-eager"

# Add --trust-remote-code for models that require it (e.g. InternLM, H2O Danube)
# CMD="$CMD --trust-remote-code"

echo "Starting vLLM | model: $MODEL | port: $PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Launch ─────────────────────────────────────────────────────────────────────
singularity exec \
    --env HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
    --env HF_TOKEN="$HF_TOKEN" \
    --env HF_HOME="$HF_CACHE_DIR" \
    --env TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers" \
    --env HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub" \
    --env VLLM_USE_FLASH_ATTENTION=0 \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env PYTHONPATH="$PYAIRPORTS_DIR:/usr/local/lib/python3.11/dist-packages" \
    --bind "/home/$USER:/home/$USER" \
    --bind /dev/shm:/dev/shm \
    --nv \
    "$SIF" $CMD
