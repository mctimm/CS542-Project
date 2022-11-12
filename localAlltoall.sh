#!/bin/bash

#SBATCH --partition debug
#SBATCH --nodes 4
#SBATCH --ntasks 16
#SBATCH --mem 2900M
#SBATCH --time 00:10:00
#SBATCH --job-name localAlltoall4x4
#SBATCH --mail-user mtimm1984@unm.edu
#SBATCH --mail-type ALL
#SBATCH --ntasks-per-node 4

. env.sh
srun --mpi=pmi2 ./build/Alltoall > results_4x4.csv
