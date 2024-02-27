#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

cudaError_t multiplyWithCuda(int *resultMatrix, const int *matrixA, const int *matrixB, unsigned int rowsA, unsigned int sharedDim, unsigned int colsB);

__global__ void matrixMultiplication(int* resultMatrix, const int* matrixA, const int* matrixB, const int rowsA, const int sharedDim, const int colsB)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rowsA && col < colsB) {
        int sum = 0;
        for (int i = 0; i < sharedDim; i++) {
            sum += matrixA[row * sharedDim + i] * matrixB[i * colsB + col];
        }
        resultMatrix[row * colsB + col] = sum;
    }
}

int main()
{
    // Declaración e inicialización de variables (Matrices)
    int* matrixA_CPU;
    int* matrixB_CPU;
    int* resultMatrix_CPU;

    int* matrixA_GPU;
    int* matrixB_GPU;
    int* resultMatrix_GPU;

    const int rowsA = 5; // Filas de A
    const int sharedDim = 5; // Columnas de A y filas de B
    const int colsB = 5; // Columnas de B

    const int dataSizeMatrixA = rowsA * sharedDim * sizeof(int);
    const int dataSizeMatrixB = sharedDim * colsB * sizeof(int);
    const int dataSizeResultMatrix = rowsA * colsB * sizeof(int);
  
    // Inicialización de las dimensiones del bloque y la cuadrícula
    dim3 blockSize(5, 5, 1);
    dim3 gridSize(1, 1, 1);

    // Asignación de memoria en CPU
    matrixA_CPU = (int*)malloc(dataSizeMatrixA);
    matrixB_CPU = (int*)malloc(dataSizeMatrixB);
    resultMatrix_CPU = (int*)malloc(dataSizeResultMatrix);

    // Asignación de memoria en GPU
    cudaMalloc((int**)&matrixA_GPU, dataSizeMatrixA);
    cudaMalloc((int**)&matrixB_GPU, dataSizeMatrixB);
    cudaMalloc((int**)&resultMatrix_GPU, dataSizeResultMatrix);

    // Inicialización de las matrices A y B con valores aleatorios
    for (int i = 0; i < rowsA * sharedDim; ++i) {
        matrixA_CPU[i] = rand() % 10;
    }
    for (int i = 0; i < sharedDim * colsB; ++i) {
        matrixB_CPU[i] = rand() % 10;
    }

    // Copia de datos de la CPU a la GPU
    cudaMemcpy(matrixA_GPU, matrixA_CPU, dataSizeMatrixA, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_GPU, matrixB_CPU, dataSizeMatrixB, cudaMemcpyHostToDevice);
    cudaMemcpy(resultMatrix_GPU, resultMatrix_CPU, dataSizeResultMatrix, cudaMemcpyHostToDevice);

    // Llamada al kernel
    matrixMultiplication<<<gridSize, blockSize>>>(resultMatrix_GPU, matrixA_GPU, matrixB_GPU, rowsA, sharedDim, colsB);

    // Copia de datos de la GPU a la CPU
    cudaMemcpy(resultMatrix_CPU, resultMatrix_GPU, dataSizeResultMatrix, cudaMemcpyDeviceToHost);

    // Imprimir resultados de la matriz resultante
    for (int i = 0; i < rowsA * colsB; ++i) {
        printf("resultMatrix[%d] = %d\n", i, resultMatrix_CPU[i]);
    }

    // Liberación de memoria
    cudaFree(matrixA_GPU);
    cudaFree(matrixB_GPU);
    cudaFree(resultMatrix_GPU);
    
    // Liberación de memoria en la CPU
    free(matrixA_CPU);
    free(matrixB_CPU);
    free(resultMatrix_CPU);

    return 0;
}
