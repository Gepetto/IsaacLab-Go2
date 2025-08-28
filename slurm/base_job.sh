#!/bin/bash
#SBATCH --job-name=batch_go2-nvidia                              # Job name
#SBATCH --ntasks=1                                               # Number of tasks
#SBATCH --ntasks-per-node=1                                      # One task per node
#SBATCH --gres=gpu:1                                             # 1 GPU per job
#SBATCH --constraint="rtx_a6000|nvidia_rtx_6000_ada_generation"  # Select specific GPUs
#SBATCH --cpus-per-task=10                                       # 10 CPU cores per task
#SBATCH --hint=nomultithread                                     # Physical cores only
#SBATCH --time=5:00:00                                           # Maximum execution time (HH:MM:SS). Maximum 20h
#SBATCH --output=logs/out/%x_%A_%a.out                           # Output log
#SBATCH --error=logs/err/%x_%A_%a.err                            # Error log

# Activate python venv
source ${ALL_CCFRWORK}/issac_lab/bin/activate

# Run training
set -x
python scripts/reinforcement_learning/cleanrl/train.py \
    --task=Isaac-Velocity-Flat-H1-v0 \
    --headless \
    --num_envs=4096 \
    --logger=wandb

echo "Job ${SLURM_ARRAY_TASK_ID} completed successfully"
