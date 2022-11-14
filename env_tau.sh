#!/bin/bash
module purge
module load gcc cmake
spack load openmpi
export TAU_PROFILE=1
export TAU_COMM_MATRIX=1
export TAU_TRACE=0

