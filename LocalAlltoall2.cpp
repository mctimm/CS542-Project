#include <cassert>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

// initialize_data gives all unique data.
//void initialize_data(double *data, int size, int rank) {
//  for (int i = 0; i < size; ++i) {
//    data[i] = (double)rank + i;
//  }
//}

// this one is only for 4x4 to match spreadsheet!
void initialize_data(double *data, int size, int rank) {
  for(int i = 0; i < 16;i++){
    data[i] = 16*(rank) + i;
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
    fprintf(stderr, "[DEBUG] buffer[%d]=%02x\n", i, (int)buff[i]);
  }
}

// TODO eventually use below signature
// Current simplifications:
// - data assumed to be double
// - send and recv count are the same
// - assumes number of ranks == number of processes per rank
// - 4x4 only
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
  int rank_shared;
  MPI_Comm comm_shared;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &comm_shared);
  MPI_Comm_size(comm_shared, &ppn);
  MPI_Comm_rank(comm_shared, &rank_shared);

  if (rank == 0) {
    fprintf(stderr, "[DEBUG] rank: %d, num_ranks: %d, ppn: %d\n", rank,
            num_ranks, ppn);
  }

  int num_vals = num_ranks * sendcount;
  double *sendbuf_tmp = new double[sendcount * num_ranks];

  if (rank == 1) {
    debug_print_buffer(sendbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }
  // rotate up by rank * ppn rows
  int i_rot = (rank * ppn) % num_ranks;
  for (int i = 0; i < num_ranks * sendcount; ++i) {
    memcpy(sendbuf_tmp + (i*sendcount), sendbuf + (i_rot*sendcount), sendcount * sizeof(double));
    i_rot = (i_rot + sendcount) % num_vals;
  }

  // initialize recv buff
  memcpy(recvbuf, sendbuf_tmp, num_vals * sizeof(double));

  if (rank == 1) {
    debug_print_buffer(sendbuf_tmp, num_vals);
    fprintf(stderr, "-----------------------\n");
  }
  // send every 1*ppn rows 1 process away, locally
  MPI_Request send_request;
  MPI_Request recv_request;
  int right_neighbor_shared = (rank_shared + 1) % ppn;
  int left_neighbor_shared = (rank_shared - 1) < 0 ? ppn - 1 : rank_shared - 1;
  for (int i = ppn; i < num_ranks; i += 2*ppn) {
    MPI_Isend(sendbuf_tmp + i*sendcount , ppn*sendcount, MPI_DOUBLE, right_neighbor_shared, 0, comm_shared, &send_request);
    MPI_Irecv(recvbuf + i*sendcount, ppn*sendcount, MPI_DOUBLE, left_neighbor_shared, 0, comm_shared, &recv_request);
    MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
  }
  memcpy(sendbuf_tmp, recvbuf, num_vals * sizeof(double));
  if (rank == 1) {
    debug_print_buffer(sendbuf_tmp, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // send every 2*ppn rows 2 processes away
  int right_two_shared = (rank_shared + 2) % ppn;
  int left_two_shared = (rank_shared + ppn - 2) % ppn;
  for (int i = 2*ppn; i < num_ranks; i += 2*2*ppn) {
    MPI_Isend(sendbuf_tmp + i*sendcount , 2*ppn*sendcount, MPI_DOUBLE, right_two_shared, 1, comm_shared, &send_request);
    MPI_Irecv(recvbuf + i*sendcount, 2*ppn*sendcount, MPI_DOUBLE, left_two_shared, 1, comm_shared, &recv_request);
    MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
  }
  memcpy(sendbuf_tmp, recvbuf, num_vals * sizeof(double));
  if (rank == 1) {
    debug_print_buffer(sendbuf_tmp, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // create a lookup table for rank (col) on node (row)
  // rank r on node n will be node_rank_table[n*ppn + r]
  int *node_rank_table = new int[num_ranks];
  for (int i = 0; i < num_ranks; ++i)
    node_rank_table[i] = i;

  // rank r on node n exchanges data with rank n node r
  int my_node = rank / ppn;
  int exchange_rank = node_rank_table[rank_shared*ppn + my_node];
  MPI_Isend(sendbuf_tmp, num_vals, MPI_DOUBLE, exchange_rank, 2, comm, &send_request);
  MPI_Irecv(recvbuf, num_vals, MPI_DOUBLE, exchange_rank,  2, comm, &recv_request);
  MPI_Wait(&send_request, MPI_STATUS_IGNORE);
  MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
  if (rank == 1) {
    debug_print_buffer(recvbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // rotate up by (node+1)*ppn rows
  // do rotation form recvbuf from last exchange into sendbuf_tmp
  i_rot = ((my_node+1) * ppn) % num_ranks;
  for (int i = 0; i < num_ranks * sendcount; ++i) {
    memcpy(sendbuf_tmp + (i*sendcount), recvbuf + (i_rot*sendcount), sendcount * sizeof(double));
    i_rot = (i_rot + sendcount) % num_vals;
  }
  if (rank == 1) {
    debug_print_buffer(sendbuf_tmp, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // reverse within chunks of size ppn
  // reverse fom sendbuf_tmp into recvbuf
  for (int i = 0; i < ppn; ++i) {
    int j_rev = ppn - 1;
    double *recv_chunk_start = recvbuf + i*sendcount*ppn;
    double *send_chunk_start = sendbuf_tmp + i*sendcount*ppn;
    for (int j = 0; j < ppn; ++j) {
      memcpy(recv_chunk_start + j*sendcount, send_chunk_start + j_rev*sendcount, sendcount * sizeof(double));
      j_rev -= 1;
    }
  }
  if (rank == 1) {
    debug_print_buffer(recvbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // rotate up ppn-rank_shared-1
  // do rotation form recvbuf into sendbuf_tmp
  i_rot = (ppn-rank_shared-1) % num_ranks;
  for (int i = 0; i < num_ranks * sendcount; ++i) {
    memcpy(sendbuf_tmp + (i*sendcount), recvbuf + (i_rot*sendcount), sendcount * sizeof(double));
    i_rot = (i_rot + sendcount) % num_vals;
  }
  if (rank == 1) {
    debug_print_buffer(sendbuf_tmp, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // clean up
  delete[] sendbuf_tmp;
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
