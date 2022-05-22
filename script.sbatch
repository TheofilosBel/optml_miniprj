#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time 06:00:00
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem 32768
#SBATCH --partition=gpu
#SBATCH --output train-gan-%J.out

# Load modules and environment packages
source env

# path="path-to-code"

python ./run_gan.py