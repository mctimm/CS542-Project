#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>

#define DEBUG 0
#define DEBUG_RANK 1

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


void rotate(void* recvbuf,
        int new_first_byte,
        int last_byte)
{
    char* recv_buffer = (char*)(recvbuf);
    std::rotate(recv_buffer, &(recv_buffer[new_first_byte]), &(recv_buffer[last_byte]));
}

int alltoall_bruck(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int tag = 102944;
    MPI_Request requests[2];

    char* recv_buffer = (char*)recvbuf;

    int recv_size;
    MPI_Type_size(recvtype, &recv_size);

    if (sendbuf != recvbuf)
        memcpy(recvbuf, sendbuf, recvcount*recv_size*num_procs);

    // Perform all-to-all
    int stride, ctr, group_size;
    int send_proc, recv_proc, size;
    int num_steps = log2(num_procs);
    int msg_size = recvcount*recv_size;
    int total_count = recvcount*num_procs;

    // TODO : could have only half this size
    char* contig_buf = (char*)malloc(total_count*recv_size);
    char* tmpbuf = (char*)malloc(total_count*recv_size);

    // 1. rotate local data
    if (rank)
        rotate(recv_buffer, rank*msg_size, num_procs*msg_size);

    if (rank == 0) for (int i = 0; i < total_count; i++)
        printf("%d\n", ((int*)(recvbuf))[i]);

    // 2. send to left, recv from right
    stride = 1;
    for (int i = 0; i < num_steps; i++)
    {
        if (rank == 0) printf("Step %d\n", i);
        recv_proc = rank - stride;
        if (recv_proc < 0) recv_proc += num_procs;
        send_proc = rank + stride;
        if (send_proc >= num_procs) send_proc -= num_procs;

        group_size = stride * recvcount;
        
        ctr = 0;
        for (int i = group_size; i < total_count; i += (group_size*2))
        {
            for (int j = 0; j < group_size; j++)
            {
                for (int k = 0; k < recv_size; k++)
                {
                    if (rank == 0) printf("i = %d, j = %d, k = %d\n", i, j, k);
                    contig_buf[ctr*recv_size+k] = recv_buffer[(i+j)*recv_size+k];
                }
                if (rank == 0) printf("Contigbuf[%d] = %d\n", ctr, ((int*)(contig_buf))[ctr]);
                ctr++;
            }
        }

        size = ((int)(total_count / group_size) * group_size) / 2;

        if (rank == 0) printf("Rank %d sending %d vals (%d) to %d\n", rank, size, ((int*)(contig_buf))[0], send_proc);
        MPI_Isend(contig_buf, size, recvtype, send_proc, tag, comm, &(requests[0]));
        MPI_Irecv(tmpbuf, size, recvtype, recv_proc, tag, comm, &(requests[1]));
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        ctr = 0;
        for (int i = group_size; i < total_count; i += (group_size*2))
        {
            for (int j = 0; j < group_size; j++)
            {
                for (int k = 0; k < recv_size; k++)
                {
                    recv_buffer[(i+j)*recv_size+k] = tmpbuf[ctr*recv_size+k];
                }
                ctr++;
            }
        }

            if (rank == 0) for (int i = 0; i < total_count; i++)
        printf("%d\n", ((int*)(recvbuf))[i]);

        stride *= 2;

    } 

    // 3. rotate local data
    if (rank < num_procs)
        rotate(recv_buffer, (rank+1)*msg_size, num_procs*msg_size);

    if (rank == 0) for (int i = 0; i < total_count; i++)
        printf("%d\n", ((int*)(recvbuf))[i]);

    // 4. reverse local data
    memcpy(tmpbuf, recv_buffer, recv_size);
    int i_rev = total_count - 1;
    for (int i = 0; i < total_count; ++i)
    {
        memcpy(recvbuf + recv_size*i, tmpbuf + recv_size*i_rev--, recv_size);
    }

    return 0;
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
    alltoall_bruck(data_send, chunk_size, MPI_DOUBLE, data_recv, chunk_size,
                   MPI_DOUBLE, MPI_COMM_WORLD);

    for (int i = 0; i < num_doubles; ++i)
      assert_doubles_approx_equal(check_data_recv[i], data_recv[i], 1e-5);

    // warmup and barrier before timing local version
    alltoall_bruck(data_send, chunk_size, MPI_DOUBLE, data_recv, chunk_size,
                   MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // start timer
    double start = 0;
    double end = 0;
    if (rank == 0)
      start = get_time();

    // alltoall many times
    for (int i = 0; i < num_measurements; ++i) {
      alltoall_bruck(data_send, chunk_size, MPI_DOUBLE, data_recv, chunk_size,
                     MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // stop timer and print result
    if (rank == 0) {
      end = get_time();
      // append time
      printf("%s,%d,%d,%g\n", "alltoall_bruck", num_procs, num_doubles,
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
