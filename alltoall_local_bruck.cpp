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
// void initialize_data(double *data, int size, int rank) {
//   for(int i = 0; i < 16; ++i){
//     data[i] = 16*(rank) + i;
//   }
// }

//// this one is only for 4x4 w/ 32 vals to match spreadsheet!
// void initialize_data(double *data, int size, int rank) {
//   int j = 0;
//   for(int i = 0; i < 32; i += 2){
//     data[i] = 16*(rank) + j;
//     data[i+1] = 16*(rank) + j;
//     ++j;
//   }
// }

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
    fprintf(stderr, "[ERROR] on rank %d/%d, assert double: want %g, got %g\n",
            rank, num_ranks, want, got);
  }
  assert(abs(want - got) <= tolerance);
}

void debug_print_buffer(const double *buff, int size) {
  int rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  for (int i = 0; i < size; ++i) {
    fprintf(stderr, "[DEBUG] on rank %d/%d, buffer[%d]=%02x\n", rank, num_ranks,
            i, (int)buff[i]);
  }
}

void alltoall_local_bruck(const double *sendbuf, int sendcount, double *recvbuf,
                          MPI_Comm comm) {
  int rank, num_procs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_procs);
  MPI_Status status;

  // split by node
  MPI_Comm comm_local;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &comm_local);

  int local_rank, local_num_procs;
  MPI_Comm_rank(comm_local, &local_rank);
  MPI_Comm_size(comm_local, &local_num_procs);
  int node = rank / local_num_procs; // bit of integer division for the nodes,
                                     // assuming alignment
  int num_vals = sendcount * num_procs;

  // temporary buffer so we dont mutate sendbuf
  double *tmpbuf = new double[num_vals];
  memcpy(recvbuf, sendbuf, num_vals * sizeof(double));

  // calculate number
  int localsends = local_num_procs / 2;

  // initial shift
  int startingIndex = local_rank * local_num_procs * sendcount % num_vals;
  // local sends
  int sendsize = num_vals / 2;
  double *sendBuffer = new double[sendsize];
  MPI_Request send_request, recv_request;
  for (int k = 1; k <= localsends; k *= 2) {
    int count = 0;
    for (int i = 0; i < local_num_procs / k; i++) {
      if ((i % 2) == 0)
        continue;
      for (int j = 0; j < local_num_procs * k * sendcount; j++) {
        sendBuffer[count++] =
            (recvbuf[(startingIndex + i * local_num_procs * k * sendcount + j) %
                     num_vals]);
      }
    }
    MPI_Isend(sendBuffer, sendsize, MPI_DOUBLE,
              (k + local_rank) % local_num_procs, 1234, comm_local,
              &send_request);

    MPI_Irecv(tmpbuf, sendsize, MPI_DOUBLE,
              (local_rank - k) >= 0 ? local_rank - k
                                    : local_rank - k + local_num_procs,
              1234, comm_local, &recv_request);
    MPI_Wait(&send_request, &status);
    MPI_Wait(&recv_request, &status);
    count = 0;
    for (int i = 0; i < local_num_procs / k; i++) {
      if (i % 2 == 0)
        continue;
      for (int j = 0; j < local_num_procs * k * sendcount; j++) {
        recvbuf[(startingIndex + i * local_num_procs * k * sendcount + j) %
                num_vals] = tmpbuf[count++];
      }
    }
  }

  // global send.
  int nextsend =
      local_num_procs * local_rank + node; // calculate who to send to.
  startingIndex = node * local_num_procs * sendcount % num_vals;
  MPI_Isend(recvbuf, num_vals, MPI_DOUBLE, nextsend, 1234, comm, &send_request);
  MPI_Irecv(tmpbuf, num_vals, MPI_DOUBLE, nextsend, 1234, comm, &recv_request);
  MPI_Wait(&send_request, &status);
  MPI_Wait(&recv_request, &status);
  for (int j = 0; j < num_vals; j++) {
    recvbuf[j] = (tmpbuf[j]);
  }

  // reverse and rotate data
  for (int i = 0; i < local_num_procs; i++) {
    int start = i * local_num_procs;
    int end = (1 + i) * local_num_procs - 1;
    while (start < end) {
      for (int j = 0; j < sendcount; j++) {
        double tmp = recvbuf[start * sendcount + j];
        recvbuf[start * sendcount + j] = recvbuf[end * sendcount + j];
        recvbuf[end * sendcount + j] = tmp;
      }
      start++;
      end--;
    }
  }

  // adjusting the starting index
  startingIndex += (((node + 1) * local_num_procs * sendcount) +
                    ((local_num_procs - local_rank - 1) * sendcount)) %
                   num_vals;

  // second local sends
  for (int k = 1; k <= localsends; k *= 2) {
    int count = 0;
    for (int i = 0; i < (num_vals / sendcount) / k; i++) {
      if ((i % 2) == 0)
        continue;
      for (int j = 0; j < k * sendcount; j++) {
        sendBuffer[count++] =
            recvbuf[(startingIndex + i * k * sendcount + j) % num_vals];
      }
    }
    MPI_Isend(sendBuffer, sendsize, MPI_DOUBLE,
              (local_rank - k) >= 0 ? local_rank - k
                                    : local_rank - k + local_num_procs,
              1234, comm_local, &send_request);
    MPI_Irecv(tmpbuf, sendsize, MPI_DOUBLE, (local_rank + k) % local_num_procs,
              1234, comm_local, &recv_request);
    MPI_Wait(&send_request, &status);
    MPI_Wait(&recv_request, &status);
    count = 0;
    for (int i = 0; i < (num_vals / sendcount) / k; i++) {
      if (i % 2 == 0)
        continue;
      for (int j = 0; j < k * sendcount; j++) {
        recvbuf[(startingIndex + i * k * sendcount + j) % num_vals] =
            tmpbuf[count++];
      }
    }
  }
  startingIndex = (startingIndex - local_rank * sendcount) > 0
                      ? (startingIndex - local_rank * sendcount)
                      : (startingIndex - local_rank * sendcount) + num_vals;

  // REVERSE AMONG GROUPS (NOT WITHIN)
  for (int i = 0; i < local_num_procs / 2; i++) {
    for (int j = 0; j < local_num_procs; j++) {
      for (int k = 0; k < sendcount; k++) {
        tmpbuf[(i * local_num_procs + j) * sendcount + k] =
            recvbuf[(((local_num_procs - 1 - i) * local_num_procs + j) *
                         sendcount +
                     k + startingIndex) %
                    num_vals];
        tmpbuf[((local_num_procs - 1 - i) * local_num_procs + j) * sendcount +
                k] = recvbuf[((i * local_num_procs + j) * sendcount + k +
                              startingIndex) %
                             num_vals];
      }
    }
  }

  // TRANSPOSE (ARRAY[I*PPN+J] = ARRAY[J*PPN+1])
  for (int i = 0; i < local_num_procs; i++) {
    for (int j = 0; j < local_num_procs; j++) {
      for (int k = 0; k < sendcount; k++) {
        recvbuf[(i * local_num_procs + j) * sendcount + k] =
            tmpbuf[(j * local_num_procs + i) * sendcount + k];
        recvbuf[(j * local_num_procs + i) * sendcount + k] =
            tmpbuf[(i * local_num_procs + j) * sendcount + k];
      }
    }
  }

  // put final answer into recvbuf
  //memcpy(recvbuf, tempbuf, num_vals * sizeof(double));

  MPI_Comm_free(&comm_local);
  delete[] sendBuffer;
  delete[] tmpbuf;
}

// main for testing and debugging
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  static const int num_measurements = 10;

  // print csv header
  // if (rank == 0)
  //    printf("algorithm,num_procs,num_doubles_per_proc,seconds\n");

  // outer loop to test many message sizes
  for (int i = log2(num_procs); i < 27; ++i) {
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
    alltoall_local_bruck(data_send, chunk_size, data_recv, MPI_COMM_WORLD);

    for (int i = 0; i < num_doubles; ++i)
      assert_doubles_approx_equal(check_data_recv[i], data_recv[i], 1e-5);

    // warmup and barrier before timing local version
    alltoall_local_bruck(data_send, chunk_size, data_recv, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // start timer
    double start = 0;
    double end = 0;
    if (rank == 0)
      start = get_time();

    // alltoall many times
    for (int i = 0; i < num_measurements; ++i) {
      alltoall_local_bruck(data_send, chunk_size, data_recv, MPI_COMM_WORLD);
    }

    // stop timer and print result
    if (rank == 0) {
      end = get_time();
      // append time
      printf("%s,%d,%d,%g\n", "alltoall_local_bruck", num_procs, num_doubles,
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
