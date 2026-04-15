#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --job-name=sample_multiple
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --account=rwth0934
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p c23g
#SBATCH --array=0-19   # array index for epochs, length of EPOCHS array minus 1

# activate conda
source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

export HDF5_USE_FILE_LOCKING=FALSE

module purge
module load CUDA
module load intel

mkdir -p logs

# parameters
TRAININGS_RUN="600000_no_overfitting" # from which trainings run the model file for sampling should be taken
EPOCHS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20) # specify the epochs to sample

EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}  # select epoch based on array index

N_JETS=50000
BATCH_SIZE=500
MAX_LENGTH=128
TOPK=5000

# classes to sample
classes=("TTBar" "QCD")

# paths
BASE="/hpcwork/rwth0934/hep_foundation_model"

OUTPUT_DIR="${BASE}/sampled_jets/${TRAININGS_RUN}/epoch_${EPOCH}"
mkdir -p "$OUTPUT_DIR"

# loop over classes
for CLASS in "${classes[@]}"; do
    MODEL_PATH="${BASE}/training/${TRAININGS_RUN}/checkpoints/${CLASS}_epoch_${EPOCH}.pt"
    NAME="${CLASS}"

    # run sampling
    python sample_jets.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --name "$NAME" \
        --n_jets $N_JETS \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        --topk $TOPK
done
