#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addArrays(int* a, int* b, int* c, int* d, int arraySize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < arraySize) {
        d[idx] = a[idx] + b[idx] + c[idx];
    }
}

int main()
{
    const int arraySize = 10000;

    int *a_cpu, *b_cpu, *c_cpu, *d_cpu;
    int *a_device, *b_device, *c_device, *d_device;

    // Allocate memory for host arrays
    a_cpu = (int*)malloc(arraySize * sizeof(int));
    b_cpu = (int*)malloc(arraySize * sizeof(int));
    c_cpu = (int*)malloc(arraySize * sizeof(int));
    d_cpu = (int*)malloc(arraySize * sizeof(int));

    // Initialize host arrays
    for (int i = 0; i < arraySize; i++) {
        a_cpu[i] = i;
        b_cpu[i] = i;
        c_cpu[i] = i;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&a_device, arraySize * sizeof(int));
    cudaMalloc((void**)&b_device, arraySize * sizeof(int));
    cudaMalloc((void**)&c_device, arraySize * sizeof(int));
    cudaMalloc((void**)&d_device, arraySize * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(a_device, a_cpu, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c_cpu, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(a_device, b_device, c_device, d_device, arraySize);

    // Copy data from device to host
    cudaMemcpy(d_cpu, d_device, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Vector Resultante:\n");
    for (int i = 0; i < arraySize; ++i) {
        printf("%d\n", d_cpu[i]);
    }

    // Free device memory
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
    cudaFree(d_device);

    // Free host memory
    free(a_cpu);
    free(b_cpu);
    free(c_cpu);
    free(d_cpu);

    return 0;
}
