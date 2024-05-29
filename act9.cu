#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

__global__ void stream_test(int* in, int* out, int size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
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
    int NUM_STREAMS = 10;

    // Host pointers and streams
    int** h_in, ** h_ref;
    cudaStream_t* streams;

    // Allocate memory for host pointers and streams
    h_in = (int**)malloc(NUM_STREAMS * sizeof(int*));
    h_ref = (int**)malloc(NUM_STREAMS * sizeof(int*));
    streams = (cudaStream_t*)malloc(NUM_STREAMS * sizeof(cudaStream_t));

    // Allocate host memory and create streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMallocHost((void**)&h_in[i], byte_size);
        cudaMallocHost((void**)&h_ref[i], byte_size);
        cudaStreamCreate(&streams[i]);

        // Initialize host input data
        srand((unsigned int)time(NULL) + i); // Different seed for each stream
        for (int j = 0; j < size; j++) {
            h_in[i][j] = rand();
        }
    }

    // Device pointers
    int** d_in, ** d_out;

    // Allocate memory for device pointers
    d_in = (int**)malloc(NUM_STREAMS * sizeof(int*));
    d_out = (int**)malloc(NUM_STREAMS * sizeof(int*));

    // Allocate device memory
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMalloc((void**)&d_in[i], byte_size);
        cudaMalloc((void**)&d_out[i], byte_size);
    }

    // Kernel Launch and data transfer
    dim3 block(128);
    dim3 grid(size / block.x);

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpyAsync(d_in[i], h_in[i], byte_size, cudaMemcpyHostToDevice, streams[i]);
        stream_test << <grid, block, 0, streams[i] >> > (d_in[i], d_out[i], size);
        cudaMemcpyAsync(h_ref[i], d_out[i], byte_size, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Free memory
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFreeHost(h_in[i]);
        cudaFreeHost(h_ref[i]);
        cudaFree(d_in[i]);
        cudaFree(d_out[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Free allocated arrays
    free(h_in);
    free(h_ref);
    free(d_in);
    free(d_out);
    free(streams);

    return 0;
}