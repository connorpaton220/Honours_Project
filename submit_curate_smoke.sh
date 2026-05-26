#!/bin/bash
#SBATCH --job-name=cms-smoke-curate
#SBATCH --partition=ada              # change to your partition; ada is the standard CPU partition on UCT HPC
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/curate-%j.out
#SBATCH --error=logs/curate-%j.err

# Curation runs on a CPU node with internet egress. On UCT's HPC the GPU
# nodes (l40s, a100) typically don't have outbound network — that's why
# this is a separate job.

set -euo pipefail
mkdir -p logs

# --- env -----------------------------------------------------------------
module purge
module load python/3.11   # adjust to whatever your cluster exposes

# Activate the venv you set up once on the login node:
#   python -m venv ~/venvs/afriquegemma
#   source ~/venvs/afriquegemma/bin/activate
#   pip install -r requirements.txt
source "$HOME/venvs/afriquegemma/bin/activate"

# Gemma 3 tokenizer is gated. Set this in your shell rc or here:
#   export HF_TOKEN=hf_...
: "${HF_TOKEN:?HF_TOKEN must be set; see https://huggingface.co/settings/tokens}"

# Cache datasets on scratch, not in $HOME (which has a small quota on UCT HPC).
export HF_HOME="/scratch/$USER/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"

# --- run -----------------------------------------------------------------
python curate_cms.py \
    --preset smoke \
    --output-dir "/scratch/$USER/afriquegemma_smoke"
