#!/usr/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:02:00
#SBATCH --job-name=heatmap_array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --account=rwth0934
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p c23g
#SBATCH --array=0-24

source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

module purge
module load CUDA
module load intel

TRAININGS_RUN="test_run"

BASE="/hpcwork/rwth0934/hep_foundation_model"

TRAINING_PATH="${BASE}/training/${TRAININGS_RUN}"
DATA_PATH="${BASE}"
OUTPUT_PATH="${BASE}/classification/${TRAININGS_RUN}"

SAMPLED_DIR="${BASE}/sampled_jets"

EPOCHS=(2 4 6 8 10)
N=${#EPOCHS[@]}

i=$((SLURM_ARRAY_TASK_ID / N))
j=$((SLURM_ARRAY_TASK_ID % N))

TT=${EPOCHS[$i]}
QCD=${EPOCHS[$j]}

# -----------------------------
# SWITCH HERE
# -----------------------------
TAG="sampled"     # test / train / val / sampled

python classification/heatmap_worker.py \
    --training_folder "$TRAINING_PATH" \
    --data_folder "$DATA_PATH" \
    --output "$OUTPUT_PATH" \
    --tt_epoch $TT \
    --qcd_epoch $QCD \
    --tag $TAG \
    --sampled_dir "$SAMPLED_DIR"