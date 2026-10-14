#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --cpus-per-task 16
#SBATCH -t 3:00:00
#SBATCH -J mao-detect
#SBATCH --array 0-15
#SBATCH -o logs/detect_%a.log
# Stage 2: Mao detection with Qwen3-VL-32B on every page.  16 array tasks, one GPU each,
# each takes every 16th book; ~3.3 pages/s per task, about 1 h for the corpus.
# Resumable: resubmitting skips pages already in qwen/.
cd "$SLURM_SUBMIT_DIR" && source env.sh
uv run python scripts/detect.py --shard $SLURM_ARRAY_TASK_ID --nshards $SLURM_ARRAY_TASK_COUNT
