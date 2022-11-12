#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>
#include <cassert>

// initialize_data gives all unique data.
void initialize_data(double *data, int rank) {
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

void Alltoall4x4(double* data, int size){
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
    //int node = rank/4; //bit of integer division for the nodes, assuming alignment
    int node = rank/4; //bit of integer division for the nodes, assuming alignment


    //printf("%d proc, %d local\n", rank,local_rank);
    //return;

    //initial shift
    for(int i = 0; i < local_rank*local_num_procs;i++){
        double tmp = data[size-1];
        double tmp2;
        for(int j = 2; j <= size;j++){
            tmp2 = data[size-j];
            data[size-j] = tmp;
            tmp = tmp2;
        }
        data[size-1] = tmp;
    }

    
    
    //first send
    MPI_Request send_request, recv_request;

    double* recv_data = new double[size];
    for(int i = 0; i< local_num_procs;i++){
        if((i % 2) == 0) continue;
        MPI_Isend(&(data[i*local_num_procs]), local_num_procs, MPI_DOUBLE, (1+ local_rank) % local_num_procs, 1234, 
            MPI_COMM_LOCAL,&send_request);
    
    
        MPI_Irecv(&(recv_data[i*local_num_procs]), local_num_procs, MPI_DOUBLE, 
            (local_rank - 1) >= 0 ? local_rank -1: local_rank -1 + local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,&status);
        MPI_Wait(&recv_request,&status);
        for(int j =0; j < local_num_procs;j++){
            data[i*local_num_procs + j] = (recv_data[i*local_num_procs + j]);
        }
    }
    


    
    
    //second send.
    for(int i = 0; i< local_num_procs/2;i++){
        if((i % 2) == 0) continue;
        MPI_Isend(&(data[i*local_num_procs*2]), local_num_procs*2, MPI_DOUBLE, (2 + local_rank) % local_num_procs, 1234, 
            MPI_COMM_LOCAL,&send_request);
    
    
        MPI_Irecv(&(recv_data[i*local_num_procs*2]), local_num_procs*2, MPI_DOUBLE, 
            (local_rank - 2) >= 0 ? local_rank -2: local_rank -2 + local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,&status);
        MPI_Wait(&recv_request,&status);
        for(int j =0; j < local_num_procs*2;j++){
            data[i*local_num_procs*2 + j] = (recv_data[i*local_num_procs*2 + j]);
        }
    }
    

    //global send.
    int nextsend = local_num_procs * local_rank + node; //calculate who to send to.
    MPI_Isend(data, size, MPI_DOUBLE, nextsend, 1234, 
            MPI_COMM_WORLD,&send_request);
    
    
    MPI_Irecv(recv_data, size, MPI_DOUBLE, nextsend, 1234, MPI_COMM_WORLD, &recv_request);
    MPI_Wait(&send_request,&status);
    MPI_Wait(&recv_request,&status);
    for(int j =0; j < size;j++){
        data[j] = (recv_data[j]);
    }
    
    //reverse and rotate data
    for(int i = 0; i < local_num_procs;i++){
        
        int start = i*local_num_procs;
        int end = (1+i)*local_num_procs - 1;
        while (start < end)
        {
            double tmp = data[start];
            data[start] = data[end];
            data[end] = tmp;
            start++;
            end--;
        }
        
    }
    
    for(int i = 0; i < (node+1) * local_num_procs;i++){
        double tmp = data[size-1];
        double tmp2;
        for(int j = 2; j <= size;j++){
            tmp2 = data[size-j];
            data[size-j] = tmp;
            tmp = tmp2;
        }
        data[size-1] = tmp;
    }

    
    //ROTATE UP PPN - LOCAL_RANK - 1
    for(int i = 0; i< local_num_procs - local_rank-1;i++){
        double tmp = data[size-1];
        double tmp2;
        for(int j = 2; j <= size;j++){
            tmp2 = data[size-j];
            data[size-j] = tmp;
            tmp = tmp2;
        }
        data[size-1] = tmp;
    }
    
    //SEND GROUPS OF 1 ROW TO LEFT 1 PROC, LOCALLY (P3 TO P0)
    for(int i = 0; i < size;i++){
        if((i % 2) == 0) continue;
        MPI_Isend(&(data[i]), 1, MPI_DOUBLE,  (local_rank - 1) >= 0 ? local_rank -1: local_rank -1 + local_num_procs, 1234, 
            MPI_COMM_LOCAL,&send_request);
    
    
        MPI_Irecv(&(recv_data[i]), 1, MPI_DOUBLE, 
            (local_rank + 1 )% local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,&status);
        MPI_Wait(&recv_request,&status);
        data[i] = recv_data[i];
    }

    

    //SEND GROUPS OF 2 ROWS TO LEFT 2 PROC, LOCALLY (P3 TO P1)
    for(int i = 0; i < size/2;i++){
        if((i % 2) == 0) continue;
        MPI_Isend(&(data[i*2]), 2, MPI_DOUBLE,  (local_rank - 2 )>= 0 ? local_rank -2: local_rank -2+ local_num_procs, 1234, 
            MPI_COMM_LOCAL,&send_request);
    
    
        MPI_Irecv(&(recv_data[i*2]), 2, MPI_DOUBLE, 
           (local_rank + 2 )% local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,&status);
        MPI_Wait(&recv_request,&status);
        
        data[i*2] = recv_data[i*2];
        data[i*2 + 1] = recv_data[i*2+1];
    }
    
    //ROTATE UP BY LOCAL RANK
    for(int i = 0; i< local_rank;i++){
        double tmp = data[0];
        double tmp2;
        for(int j = 0; j < size;j++){
            tmp2 = data[j];
            data[j] = tmp;
            tmp = tmp2;
        }
        data[0] = tmp;
    }
    
    //REVERSE AMONG GROUPS (NOT WITHIN)

    for(int i = 0; i < local_num_procs/2;i++){
        for(int j = 0; j < local_num_procs;j++){
            double tmp = data[i*local_num_procs+j];
            data[i*local_num_procs + j] = data[(local_num_procs-1 -i)*local_num_procs + j];
            data[(local_num_procs-1 -i)*local_num_procs + j] = tmp;
        }   
    }
    
    //TRANSPOSE (ARRAY[I*PPN+J] = ARRAY[J*PPN+1])
    for(int i = 0; i < local_num_procs;i++){
        for(int j = i; j < local_num_procs;j++){
	    double tmp =  data[i*local_num_procs+j];
            data[i*local_num_procs+j] = data[j*local_num_procs+i];
	    data[j*local_num_procs+i] = tmp;
        }
    }
    
    //That's all folks
    delete[] recv_data;
}   

//main for testing and debugging
int main(int argc, char* argv[]){
    MPI_Init(&argc,&argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    static const int num_measurements = 1000;

    // print csv header
    if (rank == 0)
        printf("algorithm,num_procs,num_doubles_per_proc,seconds\n");

    double* data = new double[16];
    double* check_data_send = new double[16];
    double* check_data_recv = new double[16];
    initialize_data(data, rank);
    initialize_data(check_data_send, rank);

    // correctness check
    MPI_Alltoall(check_data_send, 16, MPI_DOUBLE, check_data_recv, 16, MPI_DOUBLE, MPI_COMM_WORLD);
    Alltoall4x4(data,16);
    for (int i = 0; i < 16; ++i)
        assert(data[i] == check_data_recv[i]);

    // this barrier makes sure we are not timing process startup
    // and data creation overhead
    MPI_Barrier(MPI_COMM_WORLD);

    // start timer
    double start = 0;
    double end = 0;
    if (rank == 0)
        start = get_time();

    // alltoall many times
    for (int i = 0; i < num_measurements; ++i) {
        Alltoall4x4(data,16);
        for(int i = 0; i < 16;i++){
            printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
        }
    }

    // stop timer and print result
    if (rank == 0) {
        end = get_time();
        printf("%s,%d,%d,%g\n", "Alltoall4x4", 16, 16, (end - start) / num_measurements); // csv row
    }

    // barrier before timing library version
    MPI_Barrier(MPI_COMM_WORLD);

    // start timer
    if (rank == 0)
        start = get_time();

    // alltoall many times
    for (int i = 0; i < num_measurements; ++i) {
        MPI_Alltoall(check_data_send, 16, MPI_DOUBLE, check_data_recv, 16, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // stop timer and print result
    if (rank == 0) {
        end = get_time();
        printf("%s,%d,%d,%g\n", "MPI_Alltoall", 16, 16, (end - start) / num_measurements); // csv row
    }

    delete[] data;
    delete[] check_data_send;
    delete[] check_data_recv;
    MPI_Finalize();
    return 0;
}
