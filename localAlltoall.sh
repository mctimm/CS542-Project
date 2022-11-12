#!/bin/bash

#SBATCH --partition normal
#SBATCH --nodes 4
#SBATCH --ntasks 16
#SBATCH --mem 2900M
#SBATCH --time 00:10:00
#SBATCH --job-name localAlltoall4x4
#SBATCH --mail-user mtimm1984@unm.edu
#SBATCH --mail-type ALL

module load openmpi

srun --mpi=pmi2 calc_pi 1048576
