#!/bin/bash

#SBATCH --partition singleGPU
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 16
#SBATCH --mem 4G
#SBATCH --time 00:30:00
#SBATCH --job-name localAlltoall16x16

. env-xena.sh
srun --mpi=pmi2 hostname
echo 'algorithm,num_procs,num_doubles_per_proc,seconds' > results_16x16.csv
srun --mpi=pmi2 -n 256 -N 16 ./build/LocalAlltoall3 >> results_16x16.csv
srun --mpi=pmi2 -n 256 -N 16 ./build/Alltoall >> results_16x16.csv

