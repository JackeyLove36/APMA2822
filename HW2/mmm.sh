#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=12:00:00

# Specify a job name:
#SBATCH -J mmm

# Specify an output file
#SBATCH -o MyMPIJob-%j.out
#SBATCH -e MyMPIJob-%j.out

./matrixMatrixMult