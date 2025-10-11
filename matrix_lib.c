/*
Dante Honorato Navaza 2321406
Maria Laura Soares 2320467
*/

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdint.h>
#include <pthread.h>
#include "matrix_lib.h"

static int num_threads = 1;  
static float g_scalar;
static Matrix *g_matrix_scalar;

static Matrix *thread_matrixA;
static Matrix *thread_matrixB_T;
static Matrix *thread_matrixC;

void set_number_threads(int n) {
    if (n > 0)
        num_threads = n;
}

Matrix* getMatrixTransposed(Matrix* matrix) {
    if (!matrix || !matrix->rows) return NULL;

    Matrix* transposed = malloc(sizeof(Matrix));
    transposed->height = matrix->width;
    transposed->width = matrix->height;

    size_t size = transposed->height * transposed->width;
    transposed->rows = (float*)aligned_alloc(32, sizeof(float) * size);
    if (!transposed->rows)
        return NULL;

    size_t block_size = 8;
    for (size_t i = 0; i < matrix->height; i += block_size) {
        for (size_t j = 0; j < matrix->width; j += block_size) {
            for (size_t bi = 0; bi < block_size && i + bi < matrix->height; bi++) {
                for (size_t bj = 0; bj < block_size && j + bj < matrix->width; bj++) {
                    transposed->rows[(j + bj) * transposed->width + (i + bi)] =
                        matrix->rows[(i + bi) * matrix->width + (j + bj)];
                }
            }
        }
    }
    return transposed;
}

// =================== SCALAR MULTIPLICATION (THREAD) ===================
void *scalar_worker(void *arg) {
    long tid = (long)arg;
    Matrix *m = g_matrix_scalar;
    float scalar = g_scalar;

    size_t rows_per_thread = m->height / num_threads;
    size_t start_row = tid * rows_per_thread;
    size_t end_row = (tid == num_threads - 1) ? m->height : (tid + 1) * rows_per_thread;
    size_t width = m->width;

    __m256 scalar_vec = _mm256_set1_ps(scalar);

    for (size_t i = start_row; i < end_row; i++) {
        size_t offset = i * width;
        size_t j = 0;
        for (; j + 8 <= width; j += 8) {
            __m256 vec = _mm256_load_ps(&m->rows[offset + j]);
            vec = _mm256_mul_ps(vec, scalar_vec);
            _mm256_store_ps(&m->rows[offset + j], vec);
        }
        for (; j < width; j++)
            m->rows[offset + j] *= scalar;
    }

    pthread_exit(NULL);
}

int scalarMatrixMult(float scalar, Matrix* matrix) {
    if (!matrix || !matrix->rows)
        return 0;

    g_scalar = scalar;
    g_matrix_scalar = matrix;

    pthread_t threads[num_threads];

    for (long t = 0; t < num_threads; t++)
        pthread_create(&threads[t], NULL, scalar_worker, (void*)t);

    for (int t = 0; t < num_threads; t++)
        pthread_join(threads[t], NULL);

    return 1;
}

// =================== MATRIX MULTIPLICATION (THREAD) ===================
void *matrix_worker(void *arg) {
    long tid = (long)arg;

    Matrix *A = thread_matrixA;
    Matrix *B_T = thread_matrixB_T;
    Matrix *C = thread_matrixC;

    size_t AH = A->height;
    size_t AW = A->width;
    size_t BW = B_T->height;

    size_t rows_per_thread = AH / num_threads;
    size_t start_row = tid * rows_per_thread;
    size_t end_row = (tid == num_threads - 1) ? AH : (tid + 1) * rows_per_thread;

    for (size_t i = start_row; i < end_row; i++) {
        for (size_t j = 0; j < BW; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            size_t k = 0;

            for (; k + 8 <= AW; k += 8) {
                __m256 a = _mm256_load_ps(&A->rows[i * AW + k]);
                __m256 b = _mm256_load_ps(&B_T->rows[j * B_T->width + k]);
                sum_vec = _mm256_fmadd_ps(a, b, sum_vec);
            }

            __m128 low = _mm256_castps256_ps128(sum_vec);
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum = _mm_add_ps(low, high);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            float total = _mm_cvtss_f32(sum);

            for (; k < AW; k++)
                total += A->rows[i * AW + k] * B_T->rows[j * B_T->width + k];

            C->rows[i * C->width + j] = total;
        }
    }

    pthread_exit(NULL);
}

int matrixMatrixMult(Matrix* A, Matrix* B, Matrix* C) {
    if (!A || !B || !C)
        return 0;
    if (A->width != B->height)
        return 0;

    Matrix *B_T = getMatrixTransposed(B);
    if (!B_T)
        return 0;

    thread_matrixA = A;
    thread_matrixB_T = B_T;
    thread_matrixC = C;

    pthread_t threads[num_threads];
    for (long t = 0; t < num_threads; t++)
        pthread_create(&threads[t], NULL, matrix_worker, (void*)t);

    for (int t = 0; t < num_threads; t++)
        pthread_join(threads[t], NULL);

    free(B_T->rows);
    free(B_T);

    return 1;
}
