#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
//just the sends and recieves
void Alltoall4x4(double* data, int size){
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Status status;
    int next_gap;

    //split by node
    
    int rank % 4;
    int local_num_procs = 4;
    int node = rank/4; //bit of integer division for the nodes, assuming alignment


    //printf("%d proc, %d local\n", rank,local_rank);
    //return;

    //initial shift
    for(int i = 0; i < local_rank*local_num_procs;i++){
        double tmp = data[0];
        double tmp2;
        for(int j = 1; j < size;j++){
            tmp2 = data[j];
            data[j] = tmp;
            tmp = tmp2;
        }
        data[0] = tmp;
    }
    //first send
    MPI_Request send_request, recv_request;

    double* recv_data = new double[size];
    for(int i = 0; i< local_num_procs;i++){
        if((i % 2) == 0) continue;
        printf("rank %d sending to %d\n",rank, (1+ local_rank) % local_num_procs);
            
    
    
       printf("rank $d recieving from %d\n",rank, (local_rank - 1) > 0 ? local_rank -1: local_rank -1 + local_num_procs);
    }
    
    //second send.
    for(int i = 0; i< local_num_procs/2;i++){
        if((i % 2) == 0) continue;
        printf("rank %d sending to %d\n",rank, (2 + local_rank) % local_num_procs);
    
    
        printf("rank $d recieving from %d\n",rank, (local_rank - 2) > 0 ? local_rank -2: local_rank -2 + local_num_procs);
        
    }

    //global send.
    int nextsend = local_num_procs * local_rank + node; //calculate who to send to.
   printf("rank %d sending to %d\n",rank,  nextsend);
    
    
   printf("rank $d recieving from %d\n",rank);
    
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
    
    for(int i = 0; i < local_num_procs;i++){
        double tmp = data[0];
        double tmp2;
        for(int j = 1; j < size;j++){
            tmp2 = data[j];
            data[j] = tmp;
            tmp = tmp2;
        }
        data[0] = tmp;
    }
    //ROTATE UP PPN - LOCAL_RANK - 1
    for(int i = 0; i< local_num_procs - local_rank-1;i++){
        double tmp = data[0];
        double tmp2;
        for(int j = 1; j < size;j++){
            tmp2 = data[j];
            data[j] = tmp;
            tmp = tmp2;
        }
        data[0] = tmp;
    }
    //SEND GROUPS OF 1 ROW TO LEFT 1 PROC, LOCALLY (P3 TO P0)
    for(int i = 0; i < size;i++){
        if((i % 2) == 0) continue;
        printf("rank %d sending to %d\n",rank;  (local_rank - 1) > 0 ? local_rank -1: local_rank -1 + local_num_procs);
    
    
       printf("rank $d recieving from %d\n",rank,(local_rank + 1 )% local_num_procs);
    }
    //SEND GROUPS OF 2 ROWS TO LEFT 2 PROC, LOCALLY (P3 TO P1)
    for(int i = 0; i < size;i++){
        if((i % 2) == 0) continue;
        MPI_Isend(&(data[i*2]), 2, MPI_DOUBLE,  (local_rank - 2 )> 0 ? local_rank -2: local_rank -2+ local_num_procs, 1234, 
            MPI_COMM_LOCAL,&send_request);
    
    
       printf("rank $d recieving from %d\n",rank,(local_rank + 2 )% local_num_procs);
        
    }
    
}   
//main for testing and debugging
int main(int argc, char* argv[]){
    MPI_Init(&argc,&argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    double* data = new double[16];
    for(int i = 0; i < 16;i++){
        data[i] = i*(rank+1); //giving all unique data.
    }

    Alltoall4x4(data,16);
    for(int i = 0; i < 16;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
    }

    MPI_Finalize();
    return 0;
}
