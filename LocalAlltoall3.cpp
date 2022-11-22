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

void AlltoallNoShiftBuffered(double *data,int partition, double *data_temp, int size, int recv_size){
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Status status;

    //split by node
    MPI_Comm MPI_COMM_LOCAL;

    MPI_Comm_split_type(MPI_COMM_WORLD,
        MPI_COMM_TYPE_SHARED,
        rank, MPI_INFO_NULL,
        &MPI_COMM_LOCAL);

    int local_rank, local_num_procs;
    MPI_Comm_rank(MPI_COMM_LOCAL, &local_rank);
    MPI_Comm_size(MPI_COMM_LOCAL, &local_num_procs);
    int node = rank/local_num_procs; //bit of integer division for the nodes, assuming alignment


    //printf("%d proc, %d local\n", rank,local_rank);
    //return;
    //calculate number
    int localsends = local_num_procs/2;

    //initial shift
    int startingIndex = local_rank*local_num_procs*recv_size % size;
    //local sends
    int sendsize = size/2;
    double* sendBuffer = new double[sendsize];
    MPI_Request send_request, recv_request;
    for(int k = 1; k <= localsends; k*=2){ 
        int count = 0;
        for(int i = 0; i< local_num_procs/k;i++){
            if((i % 2) == 0) continue;
            for(int j = 0; j < local_num_procs*k*recv_size;j++){
                sendBuffer[count++] = (data[(startingIndex + i*local_num_procs*k*recv_size + j) % size]);
            }
            
        }
        MPI_Isend(sendBuffer, sendsize, MPI_DOUBLE, (k+ local_rank) % local_num_procs, 1234, 
            MPI_COMM_LOCAL,&send_request);
    
    
        MPI_Irecv(data_temp, sendsize, MPI_DOUBLE, 
            (local_rank - k) >= 0 ? local_rank -k: local_rank -k + local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,&status);
        MPI_Wait(&recv_request,&status);
        count = 0;
        for(int i = 0; i< local_num_procs/k;i++){
		 if(i %2 == 0) continue;
            for(int j =0; j < local_num_procs*k*recv_size;j++){
                data[(startingIndex + i*local_num_procs*k*recv_size + j) % size] = data_temp[count++];
            }
        }
    }
    
    //global send.
    int nextsend = local_num_procs * local_rank + node; //calculate who to send to.
    startingIndex = node*local_num_procs*recv_size % size;
    MPI_Isend(data, size, MPI_DOUBLE, nextsend, 1234, 
            MPI_COMM_WORLD,&send_request);
    
    
    MPI_Irecv(data_temp, size, MPI_DOUBLE, nextsend, 1234, MPI_COMM_WORLD, &recv_request);
    MPI_Wait(&send_request,&status);
    MPI_Wait(&recv_request,&status);
    for(int j =0; j < size;j++){
        data[j] = (data_temp[j]);
    }


    
    //reverse and rotate data
    for(int i = 0; i < local_num_procs;i++){
        
        int start = i*local_num_procs;
        int end = (1+i)*local_num_procs - 1;
        while (start < end)
        {
            for(int j = 0; j < recv_size;j++){
		double tmp = data[start*recv_size+j];
                data[start*recv_size+j] = data[end*recv_size+j];
		data[end*recv_size+j] = tmp;
            }
            //double tmp = data[start];
            //data[start] = data[end];
            //data[end] = tmp;
            start++;
            end--;
        }
        
    }
    

    
    startingIndex += (((node+1) * local_num_procs * recv_size) + ((local_num_procs - local_rank-1)*recv_size)) % size;
    //adjusting the starting index

    
    
    //second local sends
    for(int k = 1; k <= localsends; k*=2){
        int count = 0;
        for(int i = 0; i < (size/recv_size)/k;i++){
            if((i % 2) == 0) continue;
            for(int j = 0; j < k*recv_size;j++){
                sendBuffer[count++] = data[(startingIndex + i*k*recv_size + j) % size];
            }
        }
        MPI_Isend(sendBuffer, sendsize, MPI_DOUBLE,  (local_rank - k) >= 0 ? local_rank -k: local_rank -k + local_num_procs, 1234, 
                MPI_COMM_LOCAL,&send_request);
        MPI_Irecv(data_temp, sendsize, MPI_DOUBLE, 
                (local_rank + k )% local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,&status);
        MPI_Wait(&recv_request,&status);
        count = 0;
        for(int i = 0; i < (size/recv_size)/k;i++){
		if(i %2 == 0) continue;
            for(int j = 0; j < k*recv_size; j++){
		
                data[(startingIndex + i*k*recv_size + j) % size] = data_temp[count++];
            }

        }
    }
    startingIndex = (startingIndex - local_rank*recv_size) > 0 ? (startingIndex - local_rank*recv_size) : (startingIndex - local_rank*recv_size) + size;
    
    //REVERSE AMONG GROUPS (NOT WITHIN)
    for(int i = 0; i < local_num_procs/2;i++){
        for(int j = 0; j < local_num_procs;j++){
            for(int k = 0; k < recv_size; k++){
            data_temp[(i*local_num_procs + j)*recv_size + k] = data[(((local_num_procs-1 -i)*local_num_procs + j)*recv_size + k + startingIndex) % size];
            data_temp[((local_num_procs-1 -i)*local_num_procs + j)*recv_size + k] = data[((i*local_num_procs+j)*recv_size + k + startingIndex) % size];
            }
        }   
    }
    //TRANSPOSE (ARRAY[I*PPN+J] = ARRAY[J*PPN+1])
    for(int i = 0; i < local_num_procs;i++){
        for(int j = 0; j < local_num_procs;j++){
            for(int k = 0; k < recv_size; k++){
                data[(i*local_num_procs+j)*recv_size + k] = data_temp[(j*local_num_procs+i)*recv_size + k];
	            data[(j*local_num_procs+i)*recv_size + k] = data_temp[(i*local_num_procs+j)*recv_size + k];
            }
        }
    }
    free(sendBuffer);
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
    AlltoallNoShiftBuffered(data_send, 1, data_recv, num_doubles, chunk_size);

    for (int i = 0; i < num_doubles; ++i)
     assert_doubles_approx_equal(check_data_send[i], data_recv[i], 1e-5);

    // warmup and barrier before timing local version
    AlltoallNoShiftBuffered(data_send, 1, data_recv, num_doubles, chunk_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // start timer
    double start = 0;
    double end = 0;
    if (rank == 0)
      start = get_time();

    // alltoall many times
    for (int i = 0; i < num_measurements; ++i) {
        AlltoallNoShiftBuffered(data_send, 1, data_recv, num_doubles, chunk_size);
    }

    // stop timer and print result
    if (rank == 0) {
      end = get_time();
      // append time
      printf("%s,%d,%d,%g\n", "RSM_Alltoall", num_procs, num_doubles,
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
