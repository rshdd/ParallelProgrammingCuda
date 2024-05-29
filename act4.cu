#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <iostream>
#include <stdio.h>


using namespace std;


__global__ void no_divergence() {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float a, b;

    int warp_id = gid / 32;

    if (warp_id % 2 == 0) {
        a = 3.2;
        b = 5.6;
    }
    else {
        a = 3.1416;
        b = 6.666;
    }
}

__global__ void divergence() {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float a, b;

    if (gid % 2 == 0) {
        a = 3.2;
        b = 5.6;
    }
    else {
        a = 3.1416;
        b = 6.666;
    }
}



int main()
{
    int size = 1 << 22;
    dim3 block(128);
    dim3 grid((size * block.x - 1) / block.x);

    no_divergence << <grid, block >> > ();
    cudaDeviceSynchronize();

    divergence << <grid, block >> > ();
    cudaDeviceSynchronize();

    cudaDeviceReset();

    return 0;
}