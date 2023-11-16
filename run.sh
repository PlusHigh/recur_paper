#!/bin/bash

#SBATCH --gpus=1



# Purge existing modules
module purge


# Load required modules
module load compilers/cuda/11.8 compilers/gcc/12.2.0  anaconda/2021.11 cudnn/8.4.0.27_cuda11.x

# Activate the Conda environment
source activate desi-public

# change directory to the directory where the script is located
cd recur

# Execute the Python script
python data_prepare.py

# Deactivate the Conda environment
conda deactivate
