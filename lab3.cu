#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MATRIX_SIZE 2
#define BLOCK_SIZE 16

// Kernel para la multiplicación de matrices en CUDA
__global__ void matrixMultiply(int* a, int* b, int* c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < width; ++k) {
        sum += a[row * width + k] * b[k * width + col];
    }
    c[row * width + col] = sum;
}

// Kernel para la suma de matrices en CUDA
__global__ void matrixAddition(int* a, int* b, int* c, int size) {
    int hiloX = threadIdx.x;
    int hiloY = threadIdx.y;

    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int dimX = blockDim.x;
    int dimY = blockDim.y;

    int globalIDx = blockX * dimX + hiloX;
    int globalIDy = blockY * dimY + hiloY;

    int gId = (globalIDy * blockDim.x * gridDim.x) + globalIDx;

    c[gId] = a[gId] + b[gId];
}

// Función para inicializar una matriz con valores aleatorios
void initializeMatrix(int* matrix, int col, int row, int number) {
    for (long int i = 0; i < row * col; i++) {
            matrix[i] = number;
    }
}

void printMatrix(int* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("Valor Matriz: %d \n", matrix[i, j]);
        }
    }
}



int main() {
    // Variables para matrices en host
    int* h_A, * h_B, * h_C;
    int* h_A2, * h_B2, * h_C2;

    // Variables para matrices en device
    int* d_A, * d_B, * d_C;
    int* d_A2, * d_B2, * d_C2;
    
    // Tamaño de las matrices
    int col = 100;
    int row = 100;

    int col2 = 2;
    int row2 = 2;
    
    // Tamaño en bytes
    long int bytes = col * row * sizeof(int);
    long int bytes2 = col2 * row2 * sizeof(int);

    // Reservar memoria en el host
    h_A = (int*)malloc(bytes);
    h_B = (int*)malloc(bytes);
    h_C = (int*)malloc(bytes);

    h_A2 = (int*)malloc(bytes2);
    h_B2 = (int*)malloc(bytes2);
    h_C2 = (int*)malloc(bytes2);

    // Inicializar las matrices h_A y h_B con valores aleatorios
    initializeMatrix(h_A, row, col, 1);
    initializeMatrix(h_B, row, col, 2);

    //row, col
    initializeMatrix(h_A2, row2, col2, 1);
    initializeMatrix(h_B2, row2, col2, 2);

    // Reservar memoria en el device
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMalloc(&d_A2, bytes2);
    cudaMalloc(&d_B2, bytes2);
    cudaMalloc(&d_C2, bytes2);

    // Copiar datos desde el host al device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_A2, h_A2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, h_B2, bytes, cudaMemcpyHostToDevice);

    // Definir dimensiones de los bloques y rejillas
    //Ejercicio 1
    dim3 blockSize(1, 1, 1);
    dim3 gridSize(1, 1, 1);

    // Medir tiempo de ejecución
    auto start = std::chrono::high_resolution_clock::now();

    // Lanzar el kernel de multiplicación de matrices
    matrixAddition << <gridSize, blockSize >> > (d_A, d_B, d_C, row);
    //matrixMultiply << <gridSize, blockSize >> > (d_A2, d_B2, d_C2, 2);
    cudaDeviceSynchronize();

    // Medir tiempo de ejecución
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Tiempo de ejecución: " << duration.count() << " ms" << std::endl;

    // Manejo de errores de CUDA
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error en CUDA: " << cudaGetErrorString(error) << std::endl;
    }

    // Copiar resultados desde el device al host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_C2, d_C2, bytes, cudaMemcpyDeviceToHost);

    // Liberar memoria
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A2);
    free(h_B2);
    free(h_C2);
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);

    return 0;
}
