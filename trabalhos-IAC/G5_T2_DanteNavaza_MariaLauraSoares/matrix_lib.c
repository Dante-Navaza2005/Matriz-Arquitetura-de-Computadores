#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  
#include <stdint.h>
#include "matrix_lib.h"

Matrix* getMatrixTransposed(Matrix* matrix) {
    if (!matrix || !matrix->rows) return NULL;

    Matrix* transposed = malloc(sizeof(Matrix));
    transposed->height = matrix->width;
    transposed->width  = matrix->height;
    size_t size = transposed->height * transposed->width;
    transposed->rows = (float*) aligned_alloc(32, sizeof(float) * size);

    size_t block_size = 8; // bloco 8x8

    for (size_t i = 0; i < matrix->height; i += block_size) {
        for (size_t j = 0; j < matrix->width; j += block_size) {
            // Processa bloco 8x8
            for (size_t block_i = 0; block_i < block_size && i + block_i < matrix->height; block_i++) {
                for (size_t block_j = 0; block_j < block_size && j + block_j < matrix->width; block_j++) {
                    // Carrega 1 float de cada vez (poderia vectorizar linhas se alinhado)
                    transposed->rows[(j + block_j) * transposed->width + (i + block_i)] =
                        matrix->rows[(i + block_i) * matrix->width + (j + block_j)];
                }
            }
        }
    }

    return transposed;
}

// Multiplicação de matriz por escalar usando AVX
int scalarMatrixMult(float scalar, Matrix* matrix) {
    if (!matrix || !matrix->rows) return 0;

    size_t size = matrix->height * matrix->width;
    __m256 scalar_vector = _mm256_set1_ps(scalar);

    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 m = _mm256_load_ps(&matrix->rows[i]);
        m = _mm256_mul_ps(m, scalar_vector);
        _mm256_store_ps(&matrix->rows[i], m);
    }
    for (; i < size; i++){
        matrix->rows[i] *= scalar;
    } 

    return 1;
}

// Multiplicação de matrizes usando AVX + FMA com B transposta
int matrixMatrixMult(Matrix* matrixA, Matrix* matrixB, Matrix* matrixC) {
    if (matrixA == NULL || matrixB == NULL || matrixC == NULL ||
        matrixA->width != matrixB->height ||
        matrixC->height != matrixA->height ||
        matrixC->width != matrixB->width) {
        return 0; 
    }

    Matrix* B_transposed = getMatrixTransposed(matrixB);
    
    if (!B_transposed){
        return 0;
    } 

    size_t AH = matrixA->height; 
    size_t AW = matrixA->width;   
    size_t BW = matrixB->width; 

    for (size_t i = 0; i < AH; i++) {
        for (size_t j = 0; j < BW; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            size_t k = 0;

            // Blocos de 8 floats
            for (; k + 8 <= AW; k += 8) {
                __m256 vec_a = _mm256_load_ps(&matrixA->rows[i * AW + k]);
                __m256 vec_b = _mm256_load_ps(&B_transposed->rows[j * B_transposed->width + k]);
                sum_vec = _mm256_fmadd_ps(vec_a, vec_b, sum_vec);
            }

            // Somar elementos do registrador sum_vec
            __m128 low  = _mm256_castps256_ps128(sum_vec);
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum  = _mm_add_ps(low, high);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            float total = _mm_cvtss_f32(sum);

            // Se sobrar elementos
            for (; k < AW; k++)
                total += matrixA->rows[i * AW + k] * B_transposed->rows[j * B_transposed->width + k];

            matrixC->rows[i * matrixC->width + j] = total;
        }
    }

    free(B_transposed->rows);
    free(B_transposed);
    return 1;
}
