#!/bin/bash

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=02:00:00         
#SBATCH --job-name=overfit
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=rwth0934  # Replace with your project-id or delete the line
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 #adjust is bigger dataset is loaded
#SBATCH -p c23g
#SBATCH --array=0-1   # adjust if more classes

### Program Code
#---- activate conda
source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

#---- load modules
module purge
module load CUDA
module load intel

#---- create log dir
mkdir -p logs

RUN_NAME="600000_no_overfitting_500_bs"

SCHEDULER=warmup_cosine
N_JETS=600000
N_JETS_VAL=200000
NUM_CONST=50
NUM_EPOCHS=20
BATCH_SIZE=500
BATCH_SIZE_VAL=500
LR=0.001
LR_MIN=1e-6
DROPOUT=0.0
WEIGHT_DECAY=0.00001

# # classes to train
classes=("TTBar" "QCD")

# select class based on array index
CLASS=${classes[$SLURM_ARRAY_TASK_ID]}

#subfolder of the trainingsfolder in which this run will be saved
FOLDER="${RUN_NAME}"

NAME="${CLASS}"

#specifiy trainings and validation files here
BASE="/hpcwork/rwth0934/hep_foundation_model"

TRAIN_FILE="${BASE}/preprocessed_data/${CLASS}_train_processed.h5"
VAL_FILE="${BASE}/preprocessed_data/${CLASS}_val_processed.h5"
OUTPUT_DIR="${BASE}/training/${FOLDER}"

python train.py --train_file "$TRAIN_FILE" \
                --val_file "$VAL_FILE" \
                --input_key "discretized" \
                --output_dir "$OUTPUT_DIR" \
                --name "$NAME" \
                --num_const $NUM_CONST \
                --num_epochs $NUM_EPOCHS \
                --n_jets $N_JETS \
                --n_jets_val $N_JETS_VAL \
                --batch_size $BATCH_SIZE \
                --batch_size_val $BATCH_SIZE_VAL \
                --lr $LR \
                --scheduler $SCHEDULER \
                --dropout $DROPOUT \
                --weight_decay $WEIGHT_DECAY
