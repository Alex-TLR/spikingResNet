#!/bin/bash
#SBATCH --job-name=copy_job
#SBATCH --output=copy_%j.log
#SBATCH --error=copy_%j.err
#SBATCH --time=12:00:00   # adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

# paths — EDIT THESE:
SOURCE_DIR="/d/hpc/home/dsluga/SNN/spikingResNet_my/data"
TARGET_DIR="/d/hpc/home/dsluga/SNN/spikingResNet/data"

echo "Starting copy at: $(date)"
echo "Copying from $SOURCE_DIR to $TARGET_DIR"
echo

# Run rsync (copies everything and overwrites existing files)
rsync -av "$SOURCE_DIR"/ "$TARGET_DIR"/

echo
echo "Copy completed at: $(date)"
