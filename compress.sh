#!/usr/bin/bash
#SBATCH --job-name=compress
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --tasks=1
#SBATCH --mem=64GB

#srun zip -r weights_conv.zip weights/conv
srun zip -r results.zip results/