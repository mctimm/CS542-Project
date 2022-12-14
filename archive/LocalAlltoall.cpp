#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>
#include <cassert>
#include <cmath>

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

void Alltoall4x4(double* data, double* data_temp, int size){
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

    for(int i = 0; i< local_num_procs;i++){
        if((i % 2) == 0) continue;
        MPI_Isend(&(data[i*local_num_procs]), local_num_procs, MPI_DOUBLE, (1+ local_rank) % local_num_procs, 1234,
            MPI_COMM_LOCAL,&send_request);


        MPI_Irecv(&(data_temp[i*local_num_procs]), local_num_procs, MPI_DOUBLE,
            (local_rank - 1) >= 0 ? local_rank -1: local_rank -1 + local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,&status);
        MPI_Wait(&recv_request,&status);
        for(int j =0; j < local_num_procs;j++){
            data[i*local_num_procs + j] = (data_temp[i*local_num_procs + j]);
        }
    }





    //second send.
    for(int i = 0; i< local_num_procs/2;i++){
        if((i % 2) == 0) continue;
        MPI_Isend(&(data[i*local_num_procs*2]), local_num_procs*2, MPI_DOUBLE, (2 + local_rank) % local_num_procs, 1234,
            MPI_COMM_LOCAL,&send_request);


        MPI_Irecv(&(data_temp[i*local_num_procs*2]), local_num_procs*2, MPI_DOUBLE,
            (local_rank - 2) >= 0 ? local_rank -2: local_rank -2 + local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,&status);
        MPI_Wait(&recv_request,&status);
        for(int j =0; j < local_num_procs*2;j++){
            data[i*local_num_procs*2 + j] = (data_temp[i*local_num_procs*2 + j]);
        }
    }


    //global send.
    int nextsend = local_num_procs * local_rank + node; //calculate who to send to.
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


        MPI_Irecv(&(data_temp[i]), 1, MPI_DOUBLE,
            (local_rank + 1 )% local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,&status);
        MPI_Wait(&recv_request,&status);
        data[i] = data_temp[i];
    }



    //SEND GROUPS OF 2 ROWS TO LEFT 2 PROC, LOCALLY (P3 TO P1)
    for(int i = 0; i < size/2;i++){
        if((i % 2) == 0) continue;
        MPI_Isend(&(data[i*2]), 2, MPI_DOUBLE,  (local_rank - 2 )>= 0 ? local_rank -2: local_rank -2+ local_num_procs, 1234,
            MPI_COMM_LOCAL,&send_request);


        MPI_Irecv(&(data_temp[i*2]), 2, MPI_DOUBLE,
           (local_rank + 2 )% local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
        MPI_Wait(&send_request,&status);
        MPI_Wait(&recv_request,&status);

        data[i*2] = data_temp[i*2];
        data[i*2 + 1] = data_temp[i*2+1];
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
}

void Alltoallsquare(double *data, double *data_temp, int size){
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
    //local sends

    MPI_Request send_request, recv_request;
    for(int k = 1; k <= localsends; k*=2){
	// printf("k = %d\n",k);
        for(int i = 0; i< local_num_procs/k;i++){
            if((i % 2) == 0) continue;
            MPI_Isend(&(data[i*local_num_procs*k]), local_num_procs*k, MPI_DOUBLE, (k+ local_rank) % local_num_procs, 1234,
                MPI_COMM_LOCAL,&send_request);


            MPI_Irecv(&(data_temp[i*local_num_procs*k]), local_num_procs*k, MPI_DOUBLE,
                (local_rank - k) >= 0 ? local_rank -k: local_rank -k + local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
            MPI_Wait(&send_request,&status);
            MPI_Wait(&recv_request,&status);
            for(int j =0; j < local_num_procs*k;j++){
                data[i*local_num_procs*k + j] = (data_temp[i*local_num_procs*k + j]);
            }
        }
    }
    //global send.
    int nextsend = local_num_procs * local_rank + node; //calculate who to send to.
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

    //second local sends
    for(int k = 1; k <= localsends; k*=2){
        for(int i = 0; i < size/k;i++){
            if((i % 2) == 0) continue;
            MPI_Isend(&(data[i*k]), k, MPI_DOUBLE,  (local_rank - k) >= 0 ? local_rank -k: local_rank -k + local_num_procs, 1234,
                MPI_COMM_LOCAL,&send_request);


            MPI_Irecv(&(data_temp[i*k]), k, MPI_DOUBLE,
                (local_rank + k )% local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
            MPI_Wait(&send_request,&status);
            MPI_Wait(&recv_request,&status);
            for(int j = 0; j < k; j++){
                data[i*k+j] = data_temp[i*k+j];
            }
        }
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
}


void AlltoallVarSize(double *data, double *data_temp, int size, int recv_size){
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
    for(int i = 0; i < local_rank*local_num_procs*recv_size;i++){
        double tmp = data[size-1];
        double tmp2;
        for(int j = 2; j <= size;j++){
            tmp2 = data[size-j];
            data[size-j] = tmp;
            tmp = tmp2;
        }
        data[size-1] = tmp;

    }

    //local sends

    MPI_Request send_request, recv_request;
    for(int k = 1; k <= localsends; k*=2){
	// printf("k = %d\n",k);
        for(int i = 0; i< local_num_procs/k;i++){
            if((i % 2) == 0) continue;
            MPI_Isend(&(data[i*local_num_procs*k*recv_size]), local_num_procs*k*recv_size, MPI_DOUBLE, (k+ local_rank) % local_num_procs, 1234,
                MPI_COMM_LOCAL,&send_request);


            MPI_Irecv(&(data_temp[i*local_num_procs*k*recv_size]), local_num_procs*k*recv_size, MPI_DOUBLE,
                (local_rank - k) >= 0 ? local_rank -k: local_rank -k + local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
            MPI_Wait(&send_request,&status);
            MPI_Wait(&recv_request,&status);
            for(int j =0; j < local_num_procs*k*recv_size;j++){
                data[i*local_num_procs*k*recv_size + j] = (data_temp[i*local_num_procs*k*recv_size + j]);
            }
        }
    }

    //global send.
    int nextsend = local_num_procs * local_rank + node; //calculate who to send to.
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



    for(int i = 0; i < (node+1) * local_num_procs * recv_size;i++){
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
    for(int i = 0; i< (local_num_procs - local_rank-1)*recv_size;i++){
        double tmp = data[size-1];
        double tmp2;
        for(int j = 2; j <= size;j++){
            tmp2 = data[size-j];
            data[size-j] = tmp;
            tmp = tmp2;
        }
        data[size-1] = tmp;
    }

    //second local sends
    for(int k = 1; k <= localsends; k*=2){
        for(int i = 0; i < (size/recv_size)/k;i++){
            if((i % 2) == 0) continue;
            MPI_Isend(&(data[i*k*recv_size]), k*recv_size, MPI_DOUBLE,  (local_rank - k) >= 0 ? local_rank -k: local_rank -k + local_num_procs, 1234,
                MPI_COMM_LOCAL,&send_request);


            MPI_Irecv(&(data_temp[i*k*recv_size]), k*recv_size, MPI_DOUBLE,
                (local_rank + k )% local_num_procs, 1234, MPI_COMM_LOCAL, &recv_request);
            MPI_Wait(&send_request,&status);
            MPI_Wait(&recv_request,&status);
            for(int j = 0; j < k*recv_size; j++){
                data[i*k*recv_size+j] = data_temp[i*k*recv_size+j];
            }
        }
    }


    //ROTATE UP BY LOCAL RANK
    for(int i = 0; i< local_rank * recv_size;i++){
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
            for(int k = 0; k < recv_size; k++){
            double tmp = data[(i*local_num_procs+j)*recv_size + k];
            data[(i*local_num_procs + j)*recv_size + k] = data[((local_num_procs-1 -i)*local_num_procs + j)*recv_size + k];
            data[((local_num_procs-1 -i)*local_num_procs + j)*recv_size + k] = tmp;
            }
        }
    }
    //TRANSPOSE (ARRAY[I*PPN+J] = ARRAY[J*PPN+1])
    for(int i = 0; i < local_num_procs;i++){
        for(int j = i; j < local_num_procs;j++){
            for(int k = 0; k < recv_size; k++){
	            double tmp =  data[(i*local_num_procs+j)*recv_size + k];
                data[(i*local_num_procs+j)*recv_size + k] = data[(j*local_num_procs+i)*recv_size + k];
	            data[(j*local_num_procs+i)*recv_size + k] = tmp;
            }
        }
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


void assert_doubles_approx_equal(double want, double got, double tolerance) {
    if (abs(want-got) > tolerance) {
        fprintf(stderr, "[ERROR] assert double: want %g, got %g\n", want, got);
    }
    assert(abs(want-got) <= tolerance);
}

//main for testing and debugging
int main(int argc, char* argv[]){
    MPI_Init(&argc,&argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    static const int num_measurements = 1000;

    // print csv header
    //if (rank == 0)
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

        double* data = new double[num_doubles];
        double* data_temp = new double[num_doubles];
        double* check_data_send = new double[num_doubles];
        double* check_data_recv = new double[num_doubles];
        initialize_data(data, num_doubles, rank);
        initialize_data(check_data_send, num_doubles, rank);

        // correctness check
        //MPI_Alltoall(check_data_send, chunk_size, MPI_DOUBLE, check_data_recv, chunk_size, MPI_DOUBLE, MPI_COMM_WORLD);
        //AlltoallVarSize(data, data_temp, num_doubles, chunk_size);
        //for (int i = 0; i < num_doubles; ++i)
        //    assert_doubles_approx_equal(check_data_recv[i], data[i], 1e-5);

        // warmup and barrier before timing local version
        AlltoallVarSize(data, data_temp, num_doubles, chunk_size);
        MPI_Barrier(MPI_COMM_WORLD);

        // start timer
        double start = 0;
        double end = 0;
        if (rank == 0)
            start = get_time();

        // alltoall many times
        for (int i = 0; i < num_measurements; ++i) {
            AlltoallVarSize(data, data_temp, num_doubles, chunk_size);
        }

        // stop timer and print result
        if (rank == 0) {
            end = get_time();
            // append time
            printf("%s,%d,%d,%g\n", "AlltoallVarSize", num_procs, num_doubles, (end - start) / num_measurements); // csv row
        }

        // That's all folks!
        delete[] data;
        delete[] data_temp;
        delete[] check_data_send;
        delete[] check_data_recv;
    }

    MPI_Finalize();
    return 0;
}
