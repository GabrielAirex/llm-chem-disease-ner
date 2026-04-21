# vLLM on HPC

This folder contains everything needed to serve LLMs via [vLLM](https://docs.vllm.ai) inside a Singularity container on a SLURM-managed HPC cluster.

## Files

| File | Description |
|------|-------------|
| `vllm.def` | Singularity definition file (CUDA 12.1, Python 3.11, vLLM 0.6.0) |
| `example_sbatch.sh` | Annotated SBATCH script — copy and adapt for each model |
| `llama32_chat_template.jinja` | Jinja chat template for Llama 3.2 base models |

## 1. Build the Singularity image

On a machine with root access and Singularity installed:

```bash
sudo singularity build vllm.sif vllm.def
```

Then transfer the image to the cluster:

```bash
rsync -avz --progress vllm.sif <user>@<cluster>:~/llm/
```

> The resulting `.sif` file is ~10–15 GB and is **not tracked by git** (excluded via `.gitignore`).

## 2. Adapt the SBATCH script

Copy `example_sbatch.sh` and edit the variables at the top:

```bash
cp example_sbatch.sh my-model_vllm.sh
```

| Variable | What to change |
|----------|---------------|
| `--job-name` | Descriptive name shown in `squeue` |
| `--partition` | GPU partition on your cluster (e.g. `gpu-4-a100`, `gpu-v100`) |
| `MODEL` | Hugging Face model ID (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) |
| `PORT` | Unique port per model (avoids conflicts when running multiple) |
| `SIF` | Full path to `vllm.sif` on the cluster |

**Key flags to uncomment as needed:**

```bash
--enforce-eager        # Some Mistral/older models require this
--trust-remote-code    # Required for InternLM, H2O Danube, Yi, etc.
```

### GPU memory guide

| Model size | GPU | Recommended `--max-model-len` |
|-----------|-----|-------------------------------|
| 1–3 B | V100 16 GB | 8192 |
| 7–8 B | V100 32 GB / A100 40 GB | 16384 |
| 14 B | A100 40 GB | 16384 |
| 32 B | A100 80 GB | 16384 |
| 70 B | 2× A100 80 GB (`--tensor-parallel-size 2`) | 16384 |

## 3. Submit and monitor

```bash
# Submit the job
sbatch my-model_vllm.sh

# Check job status
squeue -u $USER

# Follow logs in real time
tail -f my-model-<JOB_ID>.out

# Verify the server is ready (look for "Application startup complete")
grep "startup complete" my-model-<JOB_ID>.out
```

## 4. SSH tunnel (local access)

The vLLM server runs on a compute node inside the cluster network. To reach it from your local machine, forward the port through an SSH tunnel:

```bash
# One-shot tunnel (foreground)
ssh -N -L <local-port>:<compute-node-ip>:<remote-port> -p <ssh-port> <user>@<cluster>

# Persistent tunnel with autossh (recommended)
autossh -M 0 -N -L <local-port>:<compute-node-ip>:<remote-port> -p <ssh-port> <user>@<cluster>
```

Find the compute node IP in the job output file (`hostname -I` is logged at startup).

## 5. Connect the pipeline

Update `config/config.yaml` to point to the running model:

```yaml
llm:
  base_url: "http://localhost:<port>/v1"
  model_name: "<huggingface-model-id>"
```

Verify the server is accepting requests:

```bash
curl http://localhost:<port>/v1/models
```

## 6. HuggingFace token

Gated models (Llama, Gemma, etc.) require a token. **Never hardcode it in scripts.** Export it as an environment variable before submitting:

```bash
export HF_TOKEN=hf_...
sbatch my-model_vllm.sh
```

The SBATCH script will fail explicitly if `HF_TOKEN` is not set.

## Troubleshooting

**`pyairports` import error** — already handled by the stub created in the script.

**Flash Attention error** — `VLLM_USE_FLASH_ATTENTION=0` is set in all scripts; for persistent issues add `--enforce-eager`.

**OOM / CUDA out of memory** — reduce `--max-model-len` or `--gpu-memory-utilization`, or request more GPUs with `--tensor-parallel-size`.

**Chat template error** (`default chat template is no longer allowed`) — pass the Jinja template via `--chat-template llama32_chat_template.jinja` (only needed for non-Instruct base models).

**Model not found** — ensure `HF_TOKEN` is valid and the model exists on Hugging Face Hub.
