#!/bin/bash

#SBATCH --partition normal
#SBATCH --nodes 4
#SBATCH --ntasks 16
#SBATCH --ntasks-per-node 4
#SBATCH --mem 4G
#SBATCH --time 02:00:00
#SBATCH --job-name localAlltoall4x4

. env.sh
srun --mpi=pmi2 hostname
echo 'algorithm,num_procs,num_doubles_per_proc,seconds' > results_4x4_4.csv
srun -n 16 -N 4 ./build/Alltoall >> results_4x4_2.csv
srun -n 16 -N 4 ./build/LocalAlltoall2 >> results_4x4_2.csv
srun -n 16 -N 4 ./build/LocalAlltoall3 >> results_4x4_2.csv
srun -n 16 -N 4 ./build/LocalAlltoall2_comm >> results_4x4_2.csv
srun -n 16 -N 4 ./build/LocalAlltoall3_comm >> results_4x4_2.csv

