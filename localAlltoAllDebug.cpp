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
    MPI_Comm MPI_COMM_LOCAL;

    MPI_Comm_split(MPI_COMM_WORLD,  
        rank/4,  
        rank,  
        &MPI_COMM_LOCAL);

    int local_rank, local_num_procs;
    MPI_Comm_rank(MPI_COMM_LOCAL, &local_rank);
    MPI_Comm_size(MPI_COMM_LOCAL, &local_num_procs);
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
    if(rank == 3){
    	for(int i = 0; i < size;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
    }

    
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
    if(rank == 0){
        for(int i = 0; i < 16;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
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
    if(rank == 0){
        for(int i = 0; i < 16;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
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
    if(rank == 0){
        for(int i = 0; i < 16;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
    }


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
    if(rank == 0){
        for(int i = 0; i < 16;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
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

    if(rank == 0){
        for(int i = 0; i < 16;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
    }


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
    if(rank == 0){
        for(int i = 0; i < 16;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
    }


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

    if(rank == 1){
        for(int i = 0; i < 16;i++)
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
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
    if(rank == 1){
        for(int i = 0; i < 16;i++)
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
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
    if(rank == 1){
        for(int i = 0; i < 16;i++)
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
    }
    //REVERSE AMONG GROUPS (NOT WITHIN)

    for(int i = 0; i < local_num_procs/2;i++){
        for(int j = 0; j < local_num_procs;j++){
            double tmp = data[i*local_num_procs+j];
            data[i*local_num_procs + j] = data[(local_num_procs-1 -i)*local_num_procs + j];
            data[(local_num_procs-1 -i)*local_num_procs + j] = tmp;
        }   
    }
    if(rank == 1){
        for(int i = 0; i < 16;i++)
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
    }
    //TRANSPOSE (ARRAY[I*PPN+J] = ARRAY[J*PPN+1])
    for(int i = 0; i < local_num_procs;i++){
        for(int j = i; j < local_num_procs;j++){
	    double tmp =  data[i*local_num_procs+j];
            data[i*local_num_procs+j] = data[j*local_num_procs+i];
	    data[j*local_num_procs+i] = tmp;
        }
    }
    if(rank == 1){
        for(int i = 0; i < 16;i++)
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
    
    }
    //That's all folks
    free(recv_data);
}   

//works for square powers of 2.
void Alltoallsquare(double * data, int size, int partition){
        int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Status status;
    int next_gap;

    //split by node
    MPI_Comm MPI_COMM_LOCAL;

    MPI_Comm_split(MPI_COMM_WORLD,  
        rank/partition,  
        rank,  
        &MPI_COMM_LOCAL);

    int local_rank, local_num_procs;
    MPI_Comm_rank(MPI_COMM_LOCAL, &local_rank);
    MPI_Comm_size(MPI_COMM_LOCAL, &local_num_procs);
    int node = rank/partition; //bit of integer division for the nodes, assuming alignment


    //printf("%d proc, %d local\n", rank,local_rank);
    //return;
    //calculate number
    int localsends = local_num_procs/2;

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
    if(rank == 3){
    	for(int i = 0; i < size;i++){
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
    }

    
    }
    //local sends

    MPI_Request send_request, recv_request;
    double* recv_data = new double[size];
    for(int k = 1; k <= localsends; k*=2){ 
	printf("k = %d\n",k);
        for(int i = 0; i< local_num_procs/k;i++){
            if((i % 2) == 0) continue;
            MPI_Isend(&(data[i*local_num_procs*k]), local_num_procs*k, MPI_DOUBLE, (k+ local_rank) % local_num_procs, 1234, 
                MPI_COMM_LOCAL,&send_request);
    
    
            MPI_Irecv(&(recv_data[i*local_num_procs*k]), local_num_procs*k, MPI_DOUBLE, 
                (local_rank - k) >= 0 ? local_rank -k: local_rank -k + local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
            MPI_Wait(&send_request,&status);
            MPI_Wait(&recv_request,&status);
            for(int j =0; j < local_num_procs*k;j++){
                data[i*local_num_procs*k + j] = (recv_data[i*local_num_procs*k + j]);
            }
        }
    }
    
    if(rank == 0){
        for(int i = 0; i < size;i++){
            printf("process %d, data[%d] = %x, localsends = %d\n",rank,i,int(data[i]),localsends);
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
    if(rank == 0){
        for(int i = 0; i < size;i++){
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
    }


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
    if(rank == 0){
        for(int i = 0; i < size;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
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

    if(rank == 0){
        for(int i = 0; i < size;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
    }


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
    if(rank == 0){
        for(int i = 0; i < size;i++){
        printf("process %d, data[%d] == %e\n",rank,i,data[i]);
    }


    }
    //SEND GROUPS OF 1 ROW TO LEFT 1 PROC, LOCALLY (P3 TO P0)
    for(int k = 1; k <= localsends; k*=2){
        for(int i = 0; i < size/k;i++){
            if((i % 2) == 0) continue;
            MPI_Isend(&(data[i*k]), k, MPI_DOUBLE,  (local_rank - k) >= 0 ? local_rank -k: local_rank -k + local_num_procs, 1234, 
                MPI_COMM_LOCAL,&send_request);
    
    
            MPI_Irecv(&(recv_data[i*k]), k, MPI_DOUBLE, 
                (local_rank + k )% local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
            MPI_Wait(&send_request,&status);
            MPI_Wait(&recv_request,&status);
            for(int j = 0; j < k; j++){
                data[i*k+j] = recv_data[i*k+j];
            }
        }
    }

    if(rank == 1){
        for(int i = 0; i < size;i++)
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
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
    if(rank == 1){
        for(int i = 0; i < size;i++)
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
    }
    //REVERSE AMONG GROUPS (NOT WITHIN)

    for(int i = 0; i < local_num_procs/2;i++){
        for(int j = 0; j < local_num_procs;j++){
            double tmp = data[i*local_num_procs+j];
            data[i*local_num_procs + j] = data[(local_num_procs-1 -i)*local_num_procs + j];
            data[(local_num_procs-1 -i)*local_num_procs + j] = tmp;
        }   
    }
    if(rank == 1){
        for(int i = 0; i < size;i++)
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
    }
    //TRANSPOSE (ARRAY[I*PPN+J] = ARRAY[J*PPN+1])
    for(int i = 0; i < local_num_procs;i++){
        for(int j = i; j < local_num_procs;j++){
	    double tmp =  data[i*local_num_procs+j];
            data[i*local_num_procs+j] = data[j*local_num_procs+i];
	    data[j*local_num_procs+i] = tmp;
        }
    }
    if(rank == 1){
        for(int i = 0; i < size;i++)
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
    
    }
    //That's all folks
    free(recv_data);
}

//main for testing and debugging
int main(int argc, char* argv[]){
    MPI_Init(&argc,&argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    double* data = new double[64];
    for(int i = 0; i < 64;i++){
        data[i] = 64*(rank) + i; //giving all unique data.
    }

    Alltoallsquare(data,64,8);
    for(int i = 0; i < 64;i++){
        printf("process %d, data[%d] == %x\n",rank,i,int(data[i]));
    }

    MPI_Finalize();
    return 0;
}
