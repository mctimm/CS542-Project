#include <cassert>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define DEBUG 0
#define DEBUG_RANK 1

// initialize_data gives all unique data.
void initialize_data(double *data, int size, int rank) {
  for (int i = 0; i < size; ++i) {
    data[i] = (double)rank + i;
  }
}

//// this one is only for 4x4 to match spreadsheet!
//void initialize_data(double *data, int size, int rank) {
//  for(int i = 0; i < 16; ++i){
//    data[i] = 16*(rank) + i;
//  }
//}

//// this one is only for 4x4 w/ 32 vals to match spreadsheet!
//void initialize_data(double *data, int size, int rank) {
//  int j = 0;
//  for(int i = 0; i < 32; i += 2){
//    data[i] = 16*(rank) + j;
//    data[i+1] = 16*(rank) + j;
//    ++j;
//  }
//}

// get_time gets the wall-clock time in seconds
double get_time(void) {
  struct timeval time;
  gettimeofday(&time, NULL);
  return time.tv_sec + time.tv_usec / 1000000.0;
}

void assert_doubles_approx_equal(double want, double got, double tolerance) {
  int rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  if (abs(want - got) > tolerance) {
    fprintf(stderr, "[ERROR] on rank %d/%d, assert double: want %g, got %g\n", rank, num_ranks, want, got);
  }
  assert(abs(want - got) <= tolerance);
}

void debug_print_buffer(const double *buff, int size) {
  int rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  for (int i = 0; i < size; ++i) {
    fprintf(stderr, "[DEBUG] on rank %d/%d, buffer[%d]=%02x\n", rank, num_ranks, i, (int)buff[i]);
  }
}

// lean and mean pack for doubles with no error checking
void RSM_Pack(double *inbuf, int count, double *outbuf, int *position) {
  memcpy(outbuf + *position, inbuf, count * sizeof(double));
  *position += count;
}

// lean and mean unpack for doubles with no error checking
void RSM_Unpack(double *inbuf, int *position, double *outbuf, int count) {
  memcpy(outbuf, inbuf + *position, count * sizeof(double));
  *position += count;
}

// TODO eventually use below signature
// Current simplifications:
// - data assumed to be double
// - send and recv count are the same
// - assumes number of ranks == number of processes per rank
// - tested 4x4 only, may need tweaking for others
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

  if (DEBUG && rank == DEBUG_RANK) {
    fprintf(stderr, "[DEBUG] rank: %d, num_ranks: %d, ppn: %d\n", rank,
            num_ranks, ppn);
  }

  // some values used multiple times
  const int num_vals = num_ranks * sendcount;
  const int right_neighbor_shared = (rank_shared + 1) % ppn;
  const int left_neighbor_shared = (rank_shared - 1) < 0 ? ppn - 1 : rank_shared - 1;
  const int right_two_shared = (rank_shared + 2) % ppn;
  const int left_two_shared = (rank_shared + ppn - 2) % ppn; // left(x) === right(total-x)
  const int my_node = rank / ppn;
  const int packbuf_bytes = num_vals * sizeof(double);
  double *tmpbuf = new double[num_vals];
  double *packbuf = new double[num_vals];
  double *unpackbuf = new double[num_vals];
  MPI_Request send_request;
  MPI_Request recv_request;
  int pack_position = 0;
  int unpack_position = 0;

  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(sendbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }
  // rotate up by rank * ppn rows
  int i_rot = (rank * ppn) % num_ranks;
  for (int i = 0; i < num_ranks; ++i) {
    memcpy(tmpbuf + (i*sendcount), sendbuf + (i_rot*sendcount), sendcount * sizeof(double));
    i_rot = (i_rot + 1) % num_ranks;
  }
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(tmpbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // initialize recv buff
  memcpy(recvbuf, tmpbuf, num_vals * sizeof(double));

  ////////////////////////////////////////////////
  // send every 1*ppn rows 1 process away, locally
  ////////////////////////////////////////////////
  // pack
  pack_position = 0;
  int i = 1;
  for (int i = ppn; i < num_ranks; i += 2*ppn) {
    RSM_Pack(tmpbuf + i*sendcount, ppn*sendcount, packbuf, &pack_position);
  }
  // send and recv
  MPI_Sendrecv(packbuf, pack_position, MPI_DOUBLE, right_neighbor_shared, 0,
    unpackbuf, pack_position, MPI_DOUBLE, left_neighbor_shared, 0,
    comm_shared, MPI_STATUS_IGNORE);
  // unpack
  unpack_position = 0;
  for (int i = ppn; i < num_ranks; i += 2*ppn) {
    RSM_Unpack(unpackbuf, &unpack_position, recvbuf + i*sendcount, ppn*sendcount);
  }
  memcpy(tmpbuf, recvbuf, num_vals * sizeof(double));
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(tmpbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  /////////////////////////////////////////
  // send every 2*ppn rows 2 processes away
  /////////////////////////////////////////
  // pack
  pack_position = 0;
  for (int i = 2*ppn; i < num_ranks; i += 2*2*ppn) {
    RSM_Pack(tmpbuf + i*sendcount, 2*ppn*sendcount, packbuf, &pack_position);
  }
  // send and recv
  MPI_Sendrecv(packbuf, pack_position, MPI_DOUBLE, right_two_shared, 0,
    unpackbuf, pack_position, MPI_DOUBLE, left_two_shared, 0,
    comm_shared, MPI_STATUS_IGNORE);
  // unpack
  unpack_position = 0;
  for (int i = 2*ppn; i < num_ranks; i += 2*2*ppn) {
    RSM_Unpack(unpackbuf, &unpack_position, recvbuf + i*sendcount, 2*ppn*sendcount);
  }
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(recvbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // create a lookup table for rank (col) on node (row)
  // rank r on node n will be node_rank_table[n*ppn + r]
  int *node_rank_table = new int[num_ranks];
  for (int i = 0; i < num_ranks; ++i)
    node_rank_table[i] = i;

  // rank r on node n exchanges data with rank n node r
  int exchange_rank = node_rank_table[rank_shared*ppn + my_node];
  MPI_Sendrecv(recvbuf, num_vals, MPI_DOUBLE, exchange_rank, 0,
    tmpbuf, num_vals, MPI_DOUBLE, exchange_rank,  0, comm, MPI_STATUS_IGNORE);
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(tmpbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // rotate up by (node+1)*ppn rows
  i_rot = ((my_node+1) * ppn) % num_ranks;
  for (int i = 0; i < num_ranks; ++i) {
    memcpy(recvbuf + (i*sendcount), tmpbuf + (i_rot*sendcount), sendcount * sizeof(double));
    i_rot = (i_rot + 1) % num_ranks;
  }
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(recvbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // reverse within chunks of size ppn
  for (int i = 0; i < ppn; ++i) {
    int j_rev = ppn - 1;
    double *recv_chunk_start = tmpbuf + i*sendcount*ppn;
    double *send_chunk_start = recvbuf + i*sendcount*ppn;
    for (int j = 0; j < ppn; ++j) {
      memcpy(recv_chunk_start + j*sendcount, send_chunk_start + j_rev*sendcount, sendcount * sizeof(double));
      j_rev -= 1;
    }
  }
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(tmpbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // rotate up ppn-rank_shared-1
  i_rot = (ppn-rank_shared-1) % num_ranks;
  if (DEBUG && rank == DEBUG_RANK) {
    fprintf(stderr, "rotate up ppn-rank_shared-1\n");
  }
  for (int i = 0; i < num_ranks; ++i) {
    memcpy(recvbuf + (i*sendcount), tmpbuf + (i_rot*sendcount), sendcount * sizeof(double));
    i_rot = (i_rot + 1) % num_ranks;
  }
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(recvbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // prep buff since new data
  memcpy(tmpbuf, recvbuf, num_vals * sizeof(double));

  ////////////////////////////////////////////
  // send groups of 1 row left 1 proc, locally
  ////////////////////////////////////////////
  // pack
  pack_position = 0;
  for (int i = 1; i < num_ranks; i += 2) {
    RSM_Pack(tmpbuf + i*sendcount, sendcount, packbuf, &pack_position);
  }
  // send and recv
  MPI_Sendrecv(packbuf, pack_position, MPI_DOUBLE, left_neighbor_shared, 0,
    unpackbuf, pack_position, MPI_DOUBLE, right_neighbor_shared, 0,
    comm_shared, MPI_STATUS_IGNORE);
  // unpack
  unpack_position = 0;
  for (int i = 1; i < num_ranks; i += 2) {
    RSM_Unpack(unpackbuf, &unpack_position, recvbuf + i*sendcount, sendcount);
  }
  memcpy(tmpbuf, recvbuf, num_vals * sizeof(double));
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(tmpbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  /////////////////////////////////////////////
  // send groups of 2 rows left 2 proc, locally
  /////////////////////////////////////////////
  // pack
  pack_position = 0;
  for (int i = 2; i < num_ranks; i += 4) {
    RSM_Pack(recvbuf + i*sendcount, 2*sendcount, packbuf, &pack_position);
  }
  // send and recv
  MPI_Sendrecv(packbuf, pack_position, MPI_DOUBLE, left_two_shared, 0,
    unpackbuf, pack_position, MPI_DOUBLE, right_two_shared, 0,
    comm_shared, MPI_STATUS_IGNORE);
  // unpack
  unpack_position = 0;
  for (int i = 2; i < num_ranks; i += 4) {
    RSM_Unpack(unpackbuf, &unpack_position, tmpbuf + i*sendcount, 2*sendcount);
  }
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(tmpbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // rotate down (left) by local rank
  // = 0 - rank_shared (w/ wrap around)
  // = (0 + num_ranks - rank_shared) % num_ranks
  i_rot = (num_ranks - rank_shared) % num_ranks;
  for (int i = 0; i < num_ranks; ++i) {
    memcpy(recvbuf + (i*sendcount), tmpbuf + (i_rot*sendcount), sendcount * sizeof(double));
    i_rot = (i_rot + 1) % num_ranks;
  }
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(recvbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // reverse among groups (not within)
  // i.e. reverse by chunks of size ppn
  int j_rev = ppn - 1;
  for (int i = 0; i < ppn; ++i) {
    double *recv_chunk_start = tmpbuf + i*sendcount*ppn;
    double *send_chunk_start = recvbuf + j_rev*sendcount*ppn;
    memcpy(recv_chunk_start, send_chunk_start, sendcount*ppn*sizeof(double));
    j_rev -= 1;
  }
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(tmpbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // transpose (array[i*ppn+j] = array[j*ppn+1])
  // transform from tmpbuf into recvbuf (final answer and return param)
  for (int i = 0; i < ppn; ++i) {
    for (int j = 0; j < ppn; ++j) {
      memcpy(recvbuf + (i*sendcount*ppn + j*sendcount), tmpbuf + (j*sendcount*ppn + i*sendcount), sendcount*sizeof(double));
    }
  }
  if (DEBUG && rank == DEBUG_RANK) {
    debug_print_buffer(recvbuf, num_vals);
    fprintf(stderr, "-----------------------\n");
  }

  // clean up
  delete[] tmpbuf;
  delete[] node_rank_table;
  delete[] packbuf;
  delete[] unpackbuf;

  // make sure all ranks fill return buffer before returning
  // MPI_Barrier(comm); // TODO I don't know if this is needed...
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
  for (int i = log2(num_procs); i < 20; ++i) {
    int num_doubles = pow(2, i);
    int chunk_size = num_doubles / num_procs;

    // skip chunk sizes that don't fit evenly
    if (num_doubles % num_procs != 0)
      continue;

    if (DEBUG && rank == DEBUG_RANK) {
      fprintf(stderr, "[DEBUG] num_doubles=%d\n", num_doubles);
      fprintf(stderr, "[DEBUG] chunk_size=%d\n", chunk_size);
      fprintf(stderr, "[DEBUG] num_procs=%d\n", num_procs);
    }

    double *data_send = new double[num_doubles];
    double *data_recv = new double[num_doubles];
    double *check_data_send = new double[num_doubles];
    double *check_data_recv = new double[num_doubles];
    initialize_data(data_send, num_doubles, rank);
    initialize_data(check_data_send, num_doubles, rank);

    // correctness check
    MPI_Alltoall(check_data_send, chunk_size, MPI_DOUBLE, check_data_recv,
        chunk_size, MPI_DOUBLE, MPI_COMM_WORLD);
    RSM_Alltoall(data_send, chunk_size, data_recv, MPI_COMM_WORLD);

    for (int i = 0; i < num_doubles; ++i)
        assert_doubles_approx_equal(check_data_recv[i], data_recv[i], 1e-5);

    // warmup and barrier before timing local version
    RSM_Alltoall(data_send, chunk_size, data_recv, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // start timer
    double start = 0;
    double end = 0;
    if (rank == 0)
      start = get_time();

    // alltoall many times
    for (int i = 0; i < num_measurements; ++i) {
        RSM_Alltoall(data_send, chunk_size, data_recv, MPI_COMM_WORLD);
    }

    // stop timer and print result
    if (rank == 0) {
      end = get_time();
      // append time
      printf("%s,%d,%d,%g\n", "RSM_Alltoall_manpack_sendrecv", num_procs, num_doubles,
          (end - start) / num_measurements); // csv row
    }

    // That's all folks!
    delete[] data_send;
    delete[] data_recv;
    delete[] check_data_send;
    delete[] check_data_recv;
  }

  MPI_Finalize();
  return 0;
}
