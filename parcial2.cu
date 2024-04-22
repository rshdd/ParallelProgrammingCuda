#include <iostream>
#include <vector>
#include <ctime>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <string>

#define N 9

__global__ void verificarSudokuKernel(const int* tablero, bool* resultado) {
    int idx = threadIdx.x;
    if (idx >= N) return;

    // Validar filas
    bool visto_fila[N] = { false };
    for (int j = 0; j < N; ++j) {
        int num = tablero[idx * N + j];
        if (num != 0) {
            if (visto_fila[num - 1]) {
                resultado[idx] = false;
                return;
            }
            visto_fila[num - 1] = true;
        }
    }

    // Validar columnas
    bool visto_columna[N] = { false };
    for (int i = 0; i < N; ++i) {
        int num = tablero[i * N + idx];
        if (num != 0) {
            if (visto_columna[num - 1]) {
                resultado[idx] = false;
                return;
            }
            visto_columna[num - 1] = true;
        }
    }

    // Validar cada 3x3
    int bloqueFila = (idx / 3) * 3;
    int bloqueColumna = (idx % 3) * 3;
    bool visto_bloque[N] = { false };
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int num = tablero[(bloqueFila + i) * N + (bloqueColumna + j)];
            if (num != 0) {
                if (visto_bloque[num - 1]) {
                    resultado[idx] = false;
                    return;
                }
                visto_bloque[num - 1] = true;
            }
        }
    }
    resultado[idx] = true;
}

bool verificarValidezSudoku(const std::vector<std::vector<int>>& tablero) {
    int* d_tablero;
    bool* d_resultado;
    bool h_resultado[N];

    cudaMalloc(&d_tablero, N * N * sizeof(int));
    cudaMalloc(&d_resultado, N * sizeof(bool));

    int tablero_plano[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            tablero_plano[i * N + j] = tablero[i][j];
        }
    }

    cudaMemcpy(d_tablero, tablero_plano, N * N * sizeof(int), cudaMemcpyHostToDevice);

    verificarSudokuKernel << <1, N >> > (d_tablero, d_resultado);

    cudaMemcpy(h_resultado, d_resultado, N * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_tablero);
    cudaFree(d_resultado);

    for (int i = 0; i < N; ++i) {
        if (!h_resultado[i]) {
            return false;
        }
    }

    return true;
}

bool resolverSudoku(std::vector<std::vector<int>>& tablero, int fila = 0, int columna = 0) {
    while (fila < 9 && tablero[fila][columna] != 0) {
        columna++;
        if (columna == 9) {
            columna = 0;
            fila++;
        }
    }
    if (fila == 9) {
        return true;
    }

    for (int num = 1; num <= 9; ++num) {
        bool esValido = true;
        for (int i = 0; i < N; ++i) {
            if (tablero[fila][i] == num) {
                esValido = false;
                break;
            }
        }

        for (int i = 0; i < N; ++i) {
            if (tablero[i][columna] == num) {
                esValido = false;
                break;
            }
        }

        int inicioFila = (fila / 3) * 3;
        int inicioColumna = (columna / 3) * 3;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (tablero[inicioFila + i][inicioColumna + j] == num) {
                    esValido = false;
                    break;
                }
            }
        }

        if (esValido) {
            tablero[fila][columna] = num;
            if (resolverSudoku(tablero, fila, columna)) {
                return true;
            }
            tablero[fila][columna] = 0;
        }
    }

    return false;
}

void imprimirTableroSudoku(const std::vector<std::vector<int>>& tablero) {

    const std::string lineaHorizontal = "+-------+-------+-------+";

    std::cout << lineaHorizontal << std::endl;

    for (int i = 0; i < N; ++i) {
        std::cout << "| ";
        for (int j = 0; j < N; ++j) {
            std::cout << (tablero[i][j] == 0 ? "." : std::to_string(tablero[i][j])) << " ";
            if (j % 3 == 2) {
                std::cout << "| ";
            }
        }

        std::cout << std::endl;
        if (i % 3 == 2) {
            std::cout << lineaHorizontal << std::endl;
        }
    }
}

int main() {
    std::vector<std::vector<int>> tablero = {
        {6, 0, 0, 0, 2, 3, 0, 7, 9},
        {0, 0, 4, 5, 8, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 5, 3, 0},
        {0, 0, 1, 0, 9, 0, 0, 2, 0},
        {9, 0, 0, 0, 0, 7, 0, 0, 5},
        {4, 0, 0, 0, 0, 5, 8, 0, 0},
        {5, 6, 0, 0, 7, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1, 6, 8, 0},
        {0, 9, 0, 0, 0, 8, 2, 0, 0}
    };

    if (!verificarValidezSudoku(tablero)) {
        std::cout << "El tablero de Sudoku no es válido.\n" << std::endl;
        return 1;
    }

    clock_t inicio = clock();

    if (resolverSudoku(tablero)) {
        clock_t fin = clock();
        double tiempo = double(fin - inicio) / CLOCKS_PER_SEC;

        std::cout << "Sudoku resuelto en " << tiempo << " segundos.\n" << std::endl;
        imprimirTableroSudoku(tablero);
    }
    else {
        std::cout << "No se pudo resolver el Sudoku.\n" << std::endl;
    }

    return 0;
}
