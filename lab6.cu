#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

using namespace std;

#define GPUErrorAssertion(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void quickSortCPU(int arr[], int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        int pi = i + 1;
        quickSortCPU(arr, low, pi - 1);
        quickSortCPU(arr, pi + 1, high);
    }
}

__device__ void device_swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}


void mergeCPU(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int* L = new int[n1];
    int* R = new int[n2];
    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];
    int i = 0;
    int j = 0;
    int k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
    delete[] L;
    delete[] R;
}

void mergeSortCPU(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSortCPU(arr, l, m);
        mergeSortCPU(arr, m + 1, r);
        mergeCPU(arr, l, m, r);
    }
}

void bubbleSortCPU(int arr[], int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
}

void bitonicMerge(int arr[], int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
            if (dir == (arr[i] > arr[i + k]))
                swap(arr[i], arr[i + k]);
        bitonicMerge(arr, low, k, dir);
        bitonicMerge(arr, low + k, k, dir);
    }
}

void bitonicSortCPU(int arr[], int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSortCPU(arr, low, k, true);
        bitonicSortCPU(arr, low + k, k, false);
        bitonicMerge(arr, low, cnt, dir);
    }
}

__device__ void quickSortGPU(int arr[], int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        int pi = i + 1;
        quickSortGPU(arr, low, pi - 1);
        quickSortGPU(arr, pi + 1, high);
    }
}

__global__ void quickSortKernel(int* data, const int n) {
    quickSortGPU(data, 0, n - 1);
}

__device__ void mergeSortGPU(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSortGPU(arr, l, m);
        mergeSortGPU(arr, m + 1, r);
        int* temp = new int[r - l + 1];
        int i = l, j = m + 1, k = 0;
        while (i <= m && j <= r) {
            if (arr[i] <= arr[j])
                temp[k++] = arr[i++];
            else
                temp[k++] = arr[j++];
        }
        while (i <= m)
            temp[k++] = arr[i++];
        while (j <= r)
            temp[k++] = arr[j++];
        for (i = l; i <= r; i++)
            arr[i] = temp[i - l];
        delete[] temp;
    }
}

__global__ void mergeSortKernel(int* data, const int n) {
    mergeSortGPU(data, 0, n - 1);
}

__device__ void bubbleSortGPU(int arr[], int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
                device_swap(arr[j], arr[j + 1]);
}

__device__ void bitonicMergeGPU(int arr[], int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
            if (dir == (arr[i] > arr[i + k]))
                device_swap(arr[i], arr[i + k]);
        bitonicMergeGPU(arr, low, k, dir);
        bitonicMergeGPU(arr, low + k, k, dir);
    }
}


__global__ void bubbleSortKernel(int* data, const int n) {
    bubbleSortGPU(data, n);
}

__global__ void bitonicSortKernel(int* data, const int n) {
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < n; i += k) {
                int dir = ((i / k) & 1) == 0;
                bitonicMergeGPU(data, i, k, dir);
            }
        }
    }
}

int main() {
    const int n = 100;
    int* data_cpu = (int*)malloc(n * sizeof(int));
    int* data_gpu;
    GPUErrorAssertion(cudaMalloc((int**)&data_gpu, n * sizeof(int)));

    // Initialize data_cpu with random values
    for (int i = 0; i < n; i++) {
        data_cpu[i] = rand() % 1000;
    }

    // Copy data_cpu to data_gpu
    GPUErrorAssertion(cudaMemcpy(data_gpu, data_cpu, n * sizeof(int), cudaMemcpyHostToDevice));

    // Timing variables
    chrono::duration<double> elapsed_cpu_quick, elapsed_cpu_merge, elapsed_cpu_bubble, elapsed_cpu_bitonic;
    chrono::duration<double> elapsed_gpu_quick, elapsed_gpu_merge, elapsed_gpu_bubble, elapsed_gpu_bitonic;

    // Quick Sort
    auto start_cpu_quick = chrono::steady_clock::now();
    quickSortCPU(data_cpu, 0, n - 1);
    auto end_cpu_quick = chrono::steady_clock::now();
    elapsed_cpu_quick = end_cpu_quick - start_cpu_quick;

    auto start_gpu_quick = chrono::steady_clock::now();
    quickSortKernel<<<1, 1>>>(data_gpu, n);
    GPUErrorAssertion(cudaDeviceSynchronize());
    auto end_gpu_quick = chrono::steady_clock::now();
    elapsed_gpu_quick = end_gpu_quick - start_gpu_quick;

    // Merge Sort
    auto start_cpu_merge = chrono::steady_clock::now();
    mergeSortCPU(data_cpu, 0, n - 1);
    auto end_cpu_merge = chrono::steady_clock::now();
    elapsed_cpu_merge = end_cpu_merge - start_cpu_merge;

    auto start_gpu_merge = chrono::steady_clock::now();
    mergeSortKernel<<<1, 1>>>(data_gpu, n);
    GPUErrorAssertion(cudaDeviceSynchronize());
    auto end_gpu_merge = chrono::steady_clock::now();
    elapsed_gpu_merge = end_gpu_merge - start_gpu_merge;

    // Bubble Sort
    auto start_cpu_bubble = chrono::steady_clock::now();
    bubbleSortCPU(data_cpu, n);
    auto end_cpu_bubble = chrono::steady_clock::now();
    elapsed_cpu_bubble = end_cpu_bubble - start_cpu_bubble;

    auto start_gpu_bubble = chrono::steady_clock::now();
    bubbleSortKernel<<<1, 1>>>(data_gpu, n);
    GPUErrorAssertion(cudaDeviceSynchronize());
    auto end_gpu_bubble = chrono::steady_clock::now();
    elapsed_gpu_bubble = end_gpu_bubble - start_gpu_bubble;

    // Bitonic Sort
    auto start_cpu_bitonic = chrono::steady_clock::now();
    bitonicSortCPU(data_cpu, 0, n, true);
    auto end_cpu_bitonic = chrono::steady_clock::now();
    elapsed_cpu_bitonic = end_cpu_bitonic - start_cpu_bitonic;

    auto start_gpu_bitonic = chrono::steady_clock::now();
    bitonicSortKernel<<<1, 1>>>(data_gpu, n);
    GPUErrorAssertion(cudaDeviceSynchronize());
    auto end_gpu_bitonic = chrono::steady_clock::now();
    elapsed_gpu_bitonic = end_gpu_bitonic - start_gpu_bitonic;

    // Print the table
    cout << "Algorithm\tCPU Time (ms)\tGPU Time (ms)\tOptimization\n";
    cout << "----------------------------------------------------------\n";
    cout << "Quick Sort\t" << elapsed_cpu_quick.count() * 1000 << "\t\t" << elapsed_gpu_quick.count() * 1000 << "\t\t"
         << elapsed_cpu_quick.count() / elapsed_gpu_quick.count() << "x\n";
    cout << "Merge Sort\t" << elapsed_cpu_merge.count() * 1000 << "\t\t" << elapsed_gpu_merge.count() * 1000 << "\t\t"
         << elapsed_cpu_merge.count() / elapsed_gpu_merge.count() << "x\n";
    cout << "Bubble Sort\t" << elapsed_cpu_bubble.count() * 1000 << "\t\t" << elapsed_gpu_bubble.count() * 1000 << "\t\t"
         << elapsed_cpu_bubble.count() / elapsed_gpu_bubble.count() << "x\n";
    cout << "Bitonic Sort\t" << elapsed_cpu_bitonic.count() * 1000 << "\t\t" << elapsed_gpu_bitonic.count() * 1000 << "\t\t"
         << elapsed_cpu_bitonic.count() / elapsed_gpu_bitonic.count() << "x\n";

    // Free memory
    free(data_cpu);
    cudaFree(data_gpu);

    return 0;
}