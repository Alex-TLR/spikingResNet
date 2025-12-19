#!/usr/bin/bash
#SBATCH --job-name=compress
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --tasks=1
#SBATCH --mem=64GB

srun zip -r fmnist.zip data/fmnist
srun zip -r letters.zip data/letters
srun zip -r kmnist.zip data/kmnist
srun zip -r svhn.zip data/svhn

