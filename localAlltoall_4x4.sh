#!/bin/bash

#SBATCH --partition normal
#SBATCH --nodes 4
#SBATCH --ntasks 16
#SBATCH --ntasks-per-node 4
#SBATCH --mem 4G
#SBATCH --time 00:15:00
#SBATCH --job-name localAlltoall4x4

. env.sh
srun --mpi=pmi2 hostname
echo 'algorithm,num_procs,num_doubles_per_proc,seconds' > results_4x4.csv
srun -n 16 -N 4 ./build/LocalAlltoall >> results_4x4.csv
srun -n 16 -N 4 ./build/Alltoall >> results_4x4.csv
srun -n 16 -N 4 ./build/Alltoall2 >> results_4x4.csv
srun -n 16 -N 4 ./build/Alltoall3 >> results_4x4.csv
