cmake_minimum_required(VERSION 3.15)
project(localAlltoall
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-g -O3 -Wall -Wextra -pedantic -Werror)

find_package(MPI)
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
message("Using mpi compiler wrapper ${MPI_CXX_COMPILER}.")

add_executable(localAlltoall Localalltoall.cpp)