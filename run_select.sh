#!/bin/sh
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=10:00
#SBATCH --job-name="SelectSources"
export HF_TOKEN=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
module load python/miniconda3-py3.12
python select_sources.py