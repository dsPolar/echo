#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-00:60
#SBATCH --account comsm0018
#SBATCH --mem 64GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train_mlmc.py --mode LMC --checkpoint-path /checkpoints/LMC/
python train_mlmc.py --mode MC --checkpoint-path /checkpoints/MC/
python train_mlmc.py --mode MLMC --checkpoint-path /checkpoints/MLMC/
