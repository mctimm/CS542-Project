#!/bin/bash

#SBATCH --partition normal
#SBATCH --nodes 8
#SBATCH --ntasks 64
#SBATCH --ntasks-per-node 8
#SBATCH --mem 0
#SBATCH --time 00:30:00
#SBATCH --job-name localAlltoall8x8

. env.sh
srun --mpi=pmi2 hostname
echo 'algorithm,num_procs,num_doubles_per_proc,seconds' > results_8x8_big.csv
srun --mpi=pmi2 -n 64 -N 8 ./build/LocalAlltoall3 >> results_8x8_big.csv
srun --mpi=pmi2 -n 64 -N 8 ./build/Alltoall >> results_8x8_big.csv

