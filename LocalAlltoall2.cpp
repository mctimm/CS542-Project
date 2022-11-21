#include <cassert>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

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

void assert_doubles_approx_equal(double want, double got, double tolerance) {
  if (abs(want - got) > tolerance) {
    fprintf(stderr, "[ERROR] assert double: want %g, got %g\n", want, got);
  }
  assert(abs(want - got) <= tolerance);
}

void debug_print_buffer(const double *buff, int size) {
  for (int i = 0; i < size; ++i) {
    fprintf(stderr, "[DEBUG] buffer[%d]=%f\n", i, buff[i]);
  }
}

// TODO eventually use below signature
// Current simplifications:
// - data assumed to be double
// - send and recv count are the same
// - assumes number of ranks == number of processes per rank
//
// void RSM_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
//                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
//                   MPI_Comm comm) {
void RSM_Alltoall(const double *sendbuf, int sendcount, double *recvbuf,
                  MPI_Comm comm) {
  int rank;
  int num_ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_ranks);

  int ppn;
  MPI_Comm comm_shared;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &comm_shared);
  MPI_Comm_size(comm_shared, &ppn);

  if (rank == 0) {
    fprintf(stderr, "[DEBUG] rank: %d, num_ranks: %d, ppn: %d\n", rank,
            num_ranks, ppn);
  }

  int num_vals = num_ranks * sendcount;
  double *sendbuf_tmp = new double[sendcount * num_ranks];

  // rotate up by rank * ppn rows
  if (rank == 3)
    debug_print_buffer(sendbuf, num_vals);
  int i_rot = (rank * ppn) % num_vals;
  for (int i = 0; i < num_ranks * sendcount; ++i) {
    sendbuf_tmp[i] = sendbuf[i_rot];
    i_rot = (i_rot + 1) % num_vals;
  }
  if (rank == 3)
    debug_print_buffer(sendbuf_tmp, num_vals);

}

// main for testing and debugging
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  static const int num_measurements = 1000;

  // print csv header
  // if (rank == 0)
  //    printf("algorithm,num_procs,num_doubles_per_proc,seconds\n");

  // outer loop to test many message sizes
  for (int i = log2(num_procs); i < 15; ++i) {
    int num_doubles = pow(2, i);
    int chunk_size = num_doubles / num_procs;

    // skip chunk sizes that don't fit evenly
    if (num_doubles % num_procs != 0)
      continue;

    if (rank == 0) {
      fprintf(stderr, "[DEBUG] num_doubles=%d\n", num_doubles);
      fprintf(stderr, "[DEBUG] chunk_size=%d\n", chunk_size);
      fprintf(stderr, "[DEBUG] num_procs=%d\n", num_procs);
    }

    double *data = new double[num_doubles];
    double *data_temp = new double[num_doubles];
    double *check_data_send = new double[num_doubles];
    double *check_data_recv = new double[num_doubles];
    initialize_data(data, num_doubles, rank);
    initialize_data(check_data_send, num_doubles, rank);

    // correctness check
    MPI_Alltoall(check_data_send, chunk_size, MPI_DOUBLE, check_data_recv,
                 chunk_size, MPI_DOUBLE, MPI_COMM_WORLD);
    RSM_Alltoall(check_data_send, chunk_size, check_data_recv, MPI_COMM_WORLD);

    // for (int i = 0; i < num_doubles; ++i)
    //    assert_doubles_approx_equal(check_data_recv[i], data[i], 1e-5);

    // // warmup and barrier before timing local version
    // RSM_Alltoall(check_data_send, chunk_size, MPI_DOUBLE, check_data_recv,
    // chunk_size, MPI_DOUBLE, MPI_COMM_WORLD); MPI_Barrier(MPI_COMM_WORLD);

    // // start timer
    // double start = 0;
    // double end = 0;
    // if (rank == 0)
    //     start = get_time();

    // // alltoall many times
    // for (int i = 0; i < num_measurements; ++i) {
    //     RSM_Alltoall(check_data_send, chunk_size, MPI_DOUBLE,
    //     check_data_recv, chunk_size, MPI_DOUBLE, MPI_COMM_WORLD);
    // }

    // // stop timer and print result
    // if (rank == 0) {
    //     end = get_time();
    //     // append time
    //     printf("%s,%d,%d,%g\n", "AlltoallVarSize", num_procs, num_doubles,
    //     (end - start) / num_measurements); // csv row
    // }

    // // That's all folks!
    delete[] data;
    delete[] data_temp;
    delete[] check_data_send;
    delete[] check_data_recv;

    break; // TODO remove
  }

  MPI_Finalize();
  return 0;
}
