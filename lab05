#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define GPUErrorAssertion(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void transpose(int *input, int *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index_in = y * width + x;
        int index_out = x * height + y;
        output[index_out] = input[index_in];
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const int dataSize = width * height * sizeof(int);

    int *matrix, *transposed;
    int *matrix_gpu, *transposed_gpu;

    matrix = (int*)malloc(dataSize);
    transposed = (int*)malloc(dataSize);

    GPUErrorAssertion(cudaMalloc((void**)&matrix_gpu, dataSize));
    GPUErrorAssertion(cudaMalloc((void**)&transposed_gpu, dataSize));

    for (int i = 0; i < width * height; ++i) {
        matrix[i] = rand() % 9;
    }

    printf("VALUES BEFORE: \n");
    for (int i = 0; i < 15; ++i) {
        printf("matrix[%d] = %d\n", i, matrix[i]);
    }

    GPUErrorAssertion(cudaMemcpy(matrix_gpu, matrix, dataSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    transpose<<<gridSize, blockSize>>>(matrix_gpu, transposed_gpu, width, height);
    GPUErrorAssertion(cudaDeviceSynchronize());

    GPUErrorAssertion(cudaMemcpy(transposed, transposed_gpu, dataSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 15; ++i) {
        printf("transposed[%d] = %d\n", i, transposed[i]);
    }

    cudaFree(matrix_gpu);
    cudaFree(transposed_gpu);
    free(matrix);
    free(transposed);

    return 0;
}
