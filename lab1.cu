#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void print_global_id()
{
    int globalID = (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
    printf("[DEVICE] GlobalId: %d\n", globalID);
}

int main()
{
    dim3 blockSize(4, 2, 1);
    dim3 gridSize(2, 2, 1);
    int* c_cpu;
    int* a_cpu;
    int* b_cpu;

    int* c_device;
    int* a_device;
    int* b_device;
    const int data_count = 10000;
    const int data_size = data_count * sizeof(int);
    c_cpu = (int*)malloc(data_size);
    a_cpu = (int*)malloc(data_size);
    b_cpu = (int*)malloc(data_size);

    // memory allocation
    cudaMalloc((void**)&c_device, data_size);
    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);

    // transfer CPU host to GPU device
    cudaMemcpy(c_device, c_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);

    // launch kernel
    print_global_id << <gridSize, blockSize >> > ();

    // transfer CPU host to GPU device
    cudaMemcpy(c_cpu, c_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_cpu, a_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_device, data_size, cudaMemcpyDeviceToHost);

    cudaDeviceReset();
    cudaFree(c_device);
    cudaFree(a_device);
    cudaFree(b_device);
    return 0;
}
