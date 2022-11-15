#!/bin/bash

#SBATCH --partition normal
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 4
#SBATCH --mem 4G
#SBATCH --time 00:10:00
#SBATCH --job-name localAlltoall4x4_tau

. env_tau.sh
srun --mpi=pmi2 hostname
srun tau_exec ./build/LocalAlltoall

