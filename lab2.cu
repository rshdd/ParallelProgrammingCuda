
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>;

using namespace std;


__global__ void addArrays(int* a, int* b, int* c, int* d)
{
    int hiloX = threadIdx.x;
    int hiloY = threadIdx.y;
    int hiloZ = threadIdx.z;

    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int blockZ = blockIdx.z;

    int dimX = blockDim.x;
    int dimY = blockDim.y;
    int dimZ = blockDim.z;

    int globalIDx = blockX * dimX + hiloX;
    int globalIDy = blockY * dimY + hiloY;
    int globalIDz = blockZ * dimZ + hiloZ;

    int gId = (globalIDz * dimX * dimY) + (globalIDy * blockDim.x * gridDim.x) + globalIDx;
    d[gId] = a[gId] + b[gId] + c[gId];
}

int main()
{

    const int arraySize = 10000;

    dim3 blockSize(10, 10, 10);
    dim3 gridSize(10, 10, 2);

    int a_cpu[arraySize];
    int b_cpu[arraySize];
    int c_cpu[arraySize];

    for (int i = 0; i < arraySize; i++) {
        a_cpu[i] = i;
        b_cpu[i] = i;
        c_cpu[i] = i;
    }
    int d_cpu[arraySize];

    int* a_device;
    int* b_device;
    int* c_device;
    int* d_device;

    const int dataCount = arraySize;
    const int data_size = dataCount * sizeof(int);

    // Memory allocation
    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);
    cudaMalloc((void**)&c_device, data_size);
    cudaMalloc((void**)&d_device, data_size);

    // transfer to GPU memory
    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_device, d_cpu, data_size, cudaMemcpyHostToDevice);

    // kernel launch
    addArrays << <gridSize, blockSize >> > (a_device, b_device, c_device, d_device);

    // transfer to CPU host memory from GPU device
    cudaMemcpy(c_cpu, c_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_cpu, a_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_cpu, d_device, data_size, cudaMemcpyDeviceToHost);

    printf("\n Vector Resultante: \n");
    for (int i = 0; i < arraySize; ++i) {
        printf("%d \n ", d_cpu[i]);
    }

    cudaDeviceReset();
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
    cudaFree(d_device);

    cudaDeviceSynchronize();
    return 0;
}
