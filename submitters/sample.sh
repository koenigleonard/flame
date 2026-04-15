#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --job-name=sample_jets
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --account=rwth0934
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p c23g
#SBATCH --array=0-1   # adjust if more classes

# activate conda
source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

export HDF5_USE_FILE_LOCKING=FALSE

module purge
module load CUDA
module load intel

mkdir -p logs

# parameters
TRAININGS_RUN="test_run" # from which trainings run the model file for sampling should be taken
EPOCH=2

N_JETS=50000
BATCH_SIZE=500
MAX_LENGTH=128
TOPK=5000

# classes to sample
classes=("TTBar" "QCD")

# select class based on array index
CLASS=${classes[$SLURM_ARRAY_TASK_ID]}

# paths
BASE="/hpcwork/rwth0934/hep_foundation_model"

MODEL_PATH="${BASE}/training/${TRAININGS_RUN}/checkpoints/${CLASS}_epoch_${EPOCH}.pt"
OUTPUT_DIR="${BASE}/sampled_jets/${TRAININGS_RUN}/epoch_${EPOCH}"
NAME="${CLASS}"

mkdir -p "$OUTPUT_DIR"

# run sampling
python sample_jets.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --name "$NAME" \
    --n_jets $N_JETS \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --topk $TOPK