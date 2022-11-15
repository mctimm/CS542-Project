#!/bin/bash

#SBATCH --partition normal
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 8
#SBATCH --mem 4G
#SBATCH --time 00:10:00
#SBATCH --job-name localAlltoall8x8_tau

. env_tau.sh
srun --mpi=pmi2 ls -l /etc/slurm/slurm.conf
srun --mpi=pmi2 hostname
srun tau_exec ./build/LocalAlltoall

