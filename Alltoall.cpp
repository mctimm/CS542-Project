#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>
#include <cassert>
#include <cmath>

// initialize_data gives all unique data.
void initialize_data(double *data, int size, int rank) {
    for (int i = 0; i < size; ++i) {
        data[i] = (double)rank + i;
    }
}

// get_time gets the wall-clock time in seconds
double get_time(void) {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec + time.tv_usec / 1000000.0;
}

//main for testing and debugging
int main(int argc, char* argv[]){
    MPI_Init(&argc,&argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    static const int num_measurements = 1000;

    // print csv header
    //if (rank == 0)
    //    printf("algorithm,num_procs,num_doubles_per_proc,seconds\n");

    // outer loop to test many message sizes
    for (int i = log2(num_procs); i < 20; ++i) {
        int num_doubles = pow(2, i);
        int chunk_size = num_doubles / num_procs;

        // skip chunk sizes that don't fit evenly
        if (num_doubles % num_procs != 0)
            continue;

        double* check_data_send = new double[num_doubles];
        double* check_data_recv = new double[num_doubles];
        initialize_data(check_data_send, num_doubles, rank);

        // warmup and barrier before timing library version
        MPI_Alltoall(check_data_send, chunk_size, MPI_DOUBLE, check_data_recv, chunk_size, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // start timer
        double start = 0;
        double end = 0;
        if (rank == 0)
            start = get_time();

        // alltoall many times
        for (int i = 0; i < num_measurements; ++i) {
            MPI_Alltoall(check_data_send, chunk_size, MPI_DOUBLE, check_data_recv, chunk_size, MPI_DOUBLE, MPI_COMM_WORLD);
        }

        // stop timer and print result
        if (rank == 0) {
            end = get_time();
            printf("%s,%d,%d,%g\n", "MPI_Alltoall", num_procs, num_doubles, (end - start) / num_measurements); // csv row
        }

        // That's all folks!
        delete[] check_data_send;
        delete[] check_data_recv;
    }

    MPI_Finalize();
    return 0;
}
