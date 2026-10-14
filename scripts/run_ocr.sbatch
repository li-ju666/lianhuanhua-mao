#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --cpus-per-task 16
#SBATCH -t 4:00:00
#SBATCH -J mao-ocr
#SBATCH --array 0-15
#SBATCH -o logs/ocr_%a.log
# Stage 1: caption OCR with Qwen3-VL-8B.  16 array tasks, one GPU each, each takes every
# 16th book; ~9 pages/s per task.  Resumable: resubmitting skips pages already in ocr/.
cd "$SLURM_SUBMIT_DIR" && source env.sh
uv run python scripts/ocr.py --shard $SLURM_ARRAY_TASK_ID --nshards $SLURM_ARRAY_TASK_COUNT
