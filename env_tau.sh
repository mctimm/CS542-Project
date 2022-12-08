#!/bin/bash
module purge
module load gcc cmake
# spack load openmpi
eval `~/spack/bin/spack load --sh   openmpi` # not sure why spack stopped working normally
export TAU_PROFILE=1
export TAU_COMM_MATRIX=1
export TAU_TRACE=1

