#!/usr/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --job-name=diagonal_array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --account=rwth0934
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p c23g
#SBATCH --array=0-19

source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

module purge
module load CUDA
module load intel

TRAININGS_RUN="600000_overfitting"

BASE="/hpcwork/rwth0934/hep_foundation_model"

TRAINING_PATH="${BASE}/training/${TRAININGS_RUN}"
DATA_PATH="${BASE}"
OUTPUT_PATH="${BASE}/classification/${TRAININGS_RUN}"

SAMPLED_DIR="${BASE}/sampled_jets"

EPOCHS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
N=${#EPOCHS[@]}

i=$((SLURM_ARRAY_TASK_ID % N))

TT=${EPOCHS[$i]}
QCD=${EPOCHS[$i]}

# -----------------------------
# SWITCH HERE
# -----------------------------
TAG="test"     # test / train / val / sampled

python classification/heatmap_worker.py \
    --training_folder "$TRAINING_PATH" \
    --data_folder "$DATA_PATH" \
    --output "$OUTPUT_PATH" \
    --tt_epoch $TT \
    --qcd_epoch $QCD \
    --tag $TAG \
    --sampled_dir "$SAMPLED_DIR"