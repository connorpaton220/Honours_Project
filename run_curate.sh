#!/bin/sh
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=10:00
#SBATCH --job-name="SelectSources"
export HF_TOKEN=
module load python/miniconda3-py3.12
python /scratch/ptnonc001/scripts/curate_cms.py \
    --preset smoke \
    --tokenizer gemma3 \
    --output-dir /scratch/ptncon001/afriquegemma/cms_smoke