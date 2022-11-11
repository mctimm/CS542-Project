#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void Alltoall4x4(double* data, int size){
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Status status;
    int next_gap;

    //split by node
    MPI_Comm MPI_COMM_LOCAL;

    MPI_Comm_split_type(MPI_COMM_WORLD,  
        MPI_COMM_TYPE_SHARED,  
        rank,  
        &MPI_COMM_LOCAL)

    int local_rank, local_num_procs;
    MPI_Comm_rank(MPI_COMM_LOCAL, &local_rank);
    MPI_Comm_size(MPI_COMM_LOCAL, &local_num_procs);
    int node = num_procs/rank; //bit of integer division for the nodes, assuming alignment
    //initial shift
    for(int i = 0; i < local_rank*local_num_procs;i++){
        double tmp = data[0]
        double tmp2;
        for(int j = 1; j < size;j++){
            tmp2 = data[j]
            data[j] = tmp;
            tmp = tmp2;
        }
        data[0] = tmp;
    }
    //first send
    MPI_Status send_request, recv_request;

    double recv_data = new double[size];
    for(int i = 0; i< local_num_procs;i++){
        if((i % 2) == 0) continue;
        MPI_Isend(&(data[i*local_num_procs]), local_num_procs, MPI_DOUBLE, 1+ rank % num_procs, 1234, 
            MPI_COMM_LOCAL,&send_request);
    
    
        MPI_Irecv(&(recv_data[i*local_num_procs]), local_num_procs, MPI_DOUBLE, 
            rank - i > 0 ? rank -1: rank -1 + local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,status);
        MPI_Wait(&recv_request,status);
        for(int j =0; j < local_num_procs;j++){
            data[i*local_num_procs + j] = (recv_data[i*local_num_procs + j]);
        }
    }
    
    //second send.
    for(int i = 0; i< local_num_procs/2;i++){
        if((i % 2) == 0) continue;
        MPI_Isend(&(data[i*local_num_procs*2]), local_num_procs*2, MPI_DOUBLE, 2 + rank % num_procs, 1234, 
            MPI_COMM_LOCAL,&send_request);
    
    
        MPI_Irecv(&(recv_data[i*local_num_procs*2]), local_num_procs*2, MPI_DOUBLE, 
            rank - 2 > 0 ? rank -i: rank -2 + local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,status);
        MPI_Wait(&recv_request,status);
        for(int j =0; j < local_num_procs*2;j++){
            data[i*local_num_procs*2 + j] = (recv_data[i*local_num_procs*2 + j]);
        }
    }

    //global send.
    int nextsend = local_num_procs * local_rank + node; //calculate who to send to.
    MPI_Isend(data, size, MPI_DOUBLE, nextsend, 1234, 
            MPI_COMM_LOCAL,&send_request);
    
    
    MPI_Irecv(recv_data, nextsend, MPI_DOUBLE, next, 1234, MPI_COMM_LOCAL, &recv_request);
    MPI_Wait(&send_request,status);
    MPI_Wait(&recv_request,status);
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
            data[end] = temp;
            start++;
            end--;
        }
        
    }
    
    for(int i = 0; i < local_num_procs;i++){
        double tmp = data[0]
        double tmp2;
        for(int j = 1; j < size;j++){
            tmp2 = data[j]
            data[j] = tmp;
            tmp = tmp2;
        }
        data[0] = tmp;
    }
    //ROTATE UP PPN - LOCAL_RANK - 1
    for(int i = 0; i< local_num_procs - local_rank-1;i++){
        double tmp = data[0]
        double tmp2;
        for(int j = 1; j < size;j++){
            tmp2 = data[j]
            data[j] = tmp;
            tmp = tmp2;
        }
        data[0] = tmp;
    }
    //SEND GROUPS OF 1 ROW TO LEFT 1 PROC, LOCALLY (P3 TO P0)
    
}