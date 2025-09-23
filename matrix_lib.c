#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // cabeçalho para AVX, AVX2 e FMA
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

    // processa em blocos de 8 floats (256 bits)
    for (; i + 8 <= size; i += 8) {
        // carrega 8 floats consecutivos da matriz (load alinhado)
        __m256 m = _mm256_load_ps(&(matrix->rows[i]));

        // multiplica elemento a elemento: m = m * scalar_vec
        m = _mm256_mul_ps(m, scalar_vec);

        // armazena o resultado de volta na matriz
        _mm256_store_ps(&(matrix->rows[i]), m);
    }

    // caso tenha sobrado algum elemento 
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


    for (unsigned long int i = 0; i < M; i++) {       // percorre linhas de A
        for (unsigned long int j = 0; j < P; j++) {   // percorre colunas de B

            // acumulador vetorial: inicializa com zero
            __m256 sum_vec = _mm256_setzero_ps();

            // percorre linha de A e coluna de B em blocos de 8 floats
            for (unsigned long int k = 0; k < N; k += 8) {
                // carrega 8 elementos consecutivos da linha i de A
                __m256 a_vec = _mm256_load_ps(&matrixA->rows[i * N + k]);

                // extrai 8 elementos da coluna j de B
                // (não estão consecutivos em memória (pois é coluna/vertical), por isso o uso de 
                // um buffer temporário para juntar e depois carregar)
                alignas(32) float tempB[8];
                for (int x = 0; x < 8; x++) {
                    tempB[x] = matrixB->rows[(k + x) * P + j];
                }
                __m256 b_vec = _mm256_load_ps(tempB);

                // FMA: sum_vec = sum_vec + (a_vec * b_vec)
                // -> uma multiplicação + soma numa única instrução
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            // somar os 8 elementos do registrador YMM (sum_vec)
            alignas(32) float partial[8];
            _mm256_store_ps(partial, sum_vec);

            float total = partial[0] + partial[1] + partial[2] + partial[3] +
                          partial[4] + partial[5] + partial[6] + partial[7];

            // salva resultado na posição (i,j) de C
            matrixC->rows[i * P + j] = total;
        }
    }

    return 1; 
}
