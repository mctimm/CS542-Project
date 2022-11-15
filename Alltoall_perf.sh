#!/bin/bash
perf record -F 99 -g -o $(hostname).${RANDOM}.perf.data ./build/Alltoall
