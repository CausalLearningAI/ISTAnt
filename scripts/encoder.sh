#!/bin/bash
#
#---------------------------------------------
#example SLURM job script for single CPU/GPU
#---------------------------------------------
#
#
#SBATCH --job-name=encoder
#SBATCH --output=./logs/encoder-%j.log
#
#SBATCH --time=2:00:00 
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --partition=gpu
##SBATCH –gres=gpu:0
#
source ./venv/bin/activate
srun python ./src/run_data.py


## sbatch encoder.sh

## source ./venv/bin/activate
## srun --job-name=encoder --time=1:00:00  --ntasks=1 --mem=1G --pty bash
##srun --job-name=encoder --time=24:00:00 -N 1 --cpus-per-task 8 --mem 50G --partition visualize --gres gpu:1 --pty bash
## python ./src/run_data.py