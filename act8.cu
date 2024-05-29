#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>


__global__ void stream_test(int* in, int* out, int size) {
    int gid = blockDim.x + blockIdx.x + threadIdx.x;
    if (gid < size)
    {
        // ANY CALC
        for (int i = 0; i < 25; i++) {
            out[gid] = in[gid] + (in[gid] - 1) * (gid % 10);
        }
    }
}

int main(int argc, char** argv)
{
    int size = 1 << 18;
    int byte_size = size * sizeof(int);

    // Para poder hacer streams necesitamos hacer pinned memory
    // Initiate host pointer
    int* h_in, * h_ref, * h_in2, * h_ref2;

    cudaMallocHost((void**)&h_in, byte_size);
    cudaMallocHost((void**)&h_ref, byte_size);
    cudaMallocHost((void**)&h_in2, byte_size);
    cudaMallocHost((void**)&h_ref2, byte_size);

    srand((double)time(NULL));
    for (int i = 0; i < size; i++) {
        h_in[i] = rand();
        h_in2[i] = rand();
    }

    // Allocate device pointers
    int* d_in, * d_out, * d_in2, * d_out2;
    cudaMalloc((void**)&d_in, byte_size);
    cudaMalloc((void**)&d_out, byte_size);
    cudaMalloc((void**)&d_in2, byte_size);
    cudaMalloc((void**)&d_out2, byte_size);

    // Kernel Launch
    dim3 block(128);
    dim3 grid(size / block.x);
    cudaStream_t str, str2;
    cudaStreamCreate(&str);
    cudaStreamCreate(&str2);

    // Transfer data from host to device (assigning stream)
    cudaMemcpyAsync(d_in, h_in, byte_size, cudaMemcpyHostToDevice, str);
    // tamaño de memoria compartida, stream (__external__) __shared__
    stream_test << <grid, block, 0, str >> > (d_in, d_out, size);
    cudaMemcpyAsync(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost, str);

    // Transfer data from host to device (assigning stream)
    cudaMemcpyAsync(d_in2, h_in2, byte_size, cudaMemcpyHostToDevice, str2);
    // tamaño de memoria compartida, stream (__external__) __shared__
    stream_test << <grid, block, 0, str2 >> > (d_in2, d_out2, size);
    cudaMemcpyAsync(h_ref2, d_out2, byte_size, cudaMemcpyDeviceToHost, str2);

    return 0;
}