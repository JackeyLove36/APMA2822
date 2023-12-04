#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1 --gres-flags=enforce-binding

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 00:05:00
#SBATCH -o outputs/with_gpu.out
#SBATCH -e outputs/with_gpu.err

# Load CUDA module
module load cuda/12.2.2  gcc/10.2   

# Compile CUDA program and run
nvcc -O2 matrixVectorMult.cu -o matrixVectorMult

nsys profile --stats=true --force-overwrite=true --output=outputs/gpu_report ./matrixVectorMult
# ./matrixVectorMult