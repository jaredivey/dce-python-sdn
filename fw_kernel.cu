/*
 * /usr/local/cuda/bin/nvcc -gencode arch=compute_20,code=compute_20 -o fw_kernel.ptx -ptx fw_kernel.cu 
 */

extern "C" {
#include <math.h>

__global__ void fw(float *adj_array, int *next_array, int k, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    float check;
    float next;
    if (i < N && j < N)
    {
        check = adj_array[j * N + k] + adj_array[k * N + i];
        next = next_array[j * N + k];
    }

    __syncthreads();
    if (i == 0 || j == 0 || i > N || j > N) return; 

    if (check < adj_array[j * N + i])
    {
        adj_array[j * N + i] = check;
        next_array[j * N + i] = next;
    }
}
    
}
