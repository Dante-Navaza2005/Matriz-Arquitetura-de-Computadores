/*
Dante Honorato Navaza 2321406
Maria Laura Soares 2320467
*/

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  
#include "matrix_lib.h"

// Multiplicação de uma matriz por um escalar usando AVX
int scalarMatrixMult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0; // erro
    }

    unsigned long int size = matrix->height * matrix->width;
    unsigned long int i = 0;

    // _mm256_set1_ps: broadcast -> cria um vetor YMM (256 bits) com 8 floats todos iguais ao valor escalar fornecido.
    // Esse vetor é usado para multiplicar "em paralelo" 8 elementos da matriz.
    __m256 scalar_vec = _mm256_set1_ps(scalar_value);

    // usa em blocos de 8 floats (256 bits)
    for (; i + 8 <= size; i += 8) {
        __m256 m = _mm256_load_ps(&(matrix->rows[i]));


        m = _mm256_mul_ps(m, scalar_vec);

        // salva o resultado de volta na matriz
        _mm256_store_ps(&(matrix->rows[i]), m);
    }

    // se sobrou sobrado algum elemento 
    for (; i < size; i++) {
        matrix->rows[i] *= scalar_value;
    }

    return 1; 
}

// Multiplicação de matrizes usando AVX + FMA
int matrixMatrixMult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC) {
    if (matrixA == NULL || matrixB == NULL || matrixC == NULL ||
        matrixA->width != matrixB->height ||
        matrixC->height != matrixA->height ||
        matrixC->width != matrixB->width) {
        return 0; 
    }

    unsigned long int M = matrixA->height; 
    unsigned long int N = matrixA->width;   
    unsigned long int P = matrixB->width;    


    for (unsigned long int i = 0; i < M; i++) {      
        for (unsigned long int j = 0; j < P; j++) {  

            // acumulador vetorial: inicializa com zero
            __m256 sum_vec = _mm256_setzero_ps();

            // percorre linha de A e coluna de B em blocos de 8 floats
            for (unsigned long int k = 0; k < N; k += 8) {
                // carrega 8 elementos consecutivos da linha i de A
                __m256 a_vec = _mm256_load_ps(&matrixA->rows[i * N + k]);

                // extrai 8 elementos da coluna j de B
                float* columnsB = aligned_alloc(32, 8 * sizeof(float));
                for (int x = 0; x < 8; x++) {
                    columnsB[x] = matrixB->rows[(k + x) * P + j];
                }
                __m256 b_vec = _mm256_load_ps(columnsB);

                // FMA: sum_vec = sum_vec + (a_vec * b_vec)
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            // somar os 8 elementos do registrador YMM (sum_vec)
            float* auxiliar = aligned_alloc(32, 8 * sizeof(float));
            _mm256_store_ps(auxiliar, sum_vec);

            float total = auxiliar[0] + auxiliar[1] + auxiliar[2] + auxiliar[3] +
                          auxiliar[4] + auxiliar[5] + auxiliar[6] + auxiliar[7];

            matrixC->rows[i * P + j] = total;
        }
    }

    return 1; 
}
