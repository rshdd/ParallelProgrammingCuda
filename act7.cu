#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void unrolling2(int* input, int* temp, int size) {
	// En vez de usar 4 bloques, usamos 2.  Reduces bloques a la mitad y aun asi  accedes a los datos (orig size) but with only half 
	int tid = threadIdx.x;
	int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
	int index = BLOCK_OFFSET + tid;
	int* i_data = input + BLOCK_OFFSET;

#	// Checar que los threads que se usan son igual o menor al limite de datos Ej si uso Bloque1(32) tid 1 y Bloque2 tid1 es menor que el total de datos, puedo usar ambos bloques, por lo tanto sumo el input
	if ((index + blockDim.x) < size) {
		input[index] += input[index + blockDim.x];
	}

	__syncthreads(); // All should be here, to continue. (order in the court of law)

	// 2 Factor de desdoblamiento To sum all info and put it in index 0
	// CAMBIO PARA EVITAR DIVERGENCIA offset >=32
	for (int offset = blockDim.x / 2; offset >= 32; offset = offset / 2) {
		if (tid < offset) {
			i_data[tid] += i_data[tid + offset];
		}
		__syncthreads();
	}

	// CAMBIO PARA EVITAR DIVERGENCIA
	if (tid < 32) {
		volatile int* vsmem = i_data;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	// Return sum of everything
	if (tid == 0) {
		temp[blockIdx.x] = i_data[0];
	}
}

__global__ void unrolling4(int* input, int* temp, int size) {
	// We now use 4 blocks (chunks) per 1 block
	int tid = threadIdx.x;
	int BLOCK_OFFSET = blockIdx.x * blockDim.x * 4;
	int index = BLOCK_OFFSET + tid;
	int* i_data = input + BLOCK_OFFSET;

	// Per Grid
	if ((index * 3 * blockDim.x) < size) {
		int a1 = input[index];
		int a2 = input[index + blockDim.x];
		int a3 = input[index + blockDim.x * 2];
		int a4 = input[index + blockDim.x * 3];
	}

	__syncthreads(); // All should be here, to continue. (order in the court of law)

	// Factor de desdoblamiento To sum all info and put it in index 0
	// Per Block we sum
	for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2) {
		if (tid < offset) {
			i_data[tid] += i_data[tid + offset];
		}
		__syncthreads();
	}

	// Return sum of everything
	if (tid == 0) {
		temp[blockIdx.x] = i_data[0];
	}
}

__global__ void unrolling_complete(int* int_array, int* temp_array, int size) {
	int tid = threadIdx.x;

	// element index for this thread
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	// local data pointer
	int* i_data = int_array + blockDim.x * blockIdx.x;

	if (blockDim.x == 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];
	__syncthreads();

	if (blockDim.x == 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];
	__syncthreads();

	if (blockDim.x == 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];
	__syncthreads();

	if (blockDim.x == 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];
	__syncthreads();

	if (tid < 32) {
		volatile int* vsmem = i_data;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	// Return sum of everything
	if (tid == 0) {
		temp_array[blockIdx.x] = i_data[0];
	}
}

int main(int argc, char** argv) {

	printf("Running parallel reduction with unrolling blocks8 kernel \n");
	int data_size = 1 << 10;
	int byte_size = data_size * sizeof(int);
	int block_size = 32;
	int parallel_reduction = 2;

	int* h_input, * href;
	h_input = (int*)malloc(byte_size);

	for (int i = 0; i < data_size; i++) {
		h_input[i] = (double)(rand() % 10);
	}

	dim3 block(block_size);
	dim3 grid((data_size / block_size) / parallel_reduction);

	printf("Launch parameters -> grid: %d, block: %d \n", grid.x, block.x);

	int temp = sizeof(int) * grid.x;
	h_ref = (int*)malloc(temp);

	int* d_input, * d_temp;
	cudaMalloc((void**)&d_input, byte_size);
	cudaMalloc((void**)&d_temp, temp);

	cudaMemset(d_temp, 0, temp);
	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

	if (parallel_reduction == 2)
		unrolling2 << < grid, block >> > (d_input, d_temp, data_size);
	else
		unrolling4 << < grid, block >> > (d_input, d_temp, data_size);

	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_temp, temp, cudaMemcpyDeviceToHost);

	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++) {
		gpu_result += h_ref[i];
	}

	cudaFree(d_input);
	cudaFree(d_temp);
	free(h_input);
	free(h_ref);
}