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
static float global_scalar;      
static Matrix *global_matrix_scalar; // matriz usada na mult escalar

static Matrix *thread_matrixA;   // ponteiro global A
static Matrix *thread_matrixB_transposed; // ponteiro global B transposta
static Matrix *thread_matrixC;   // ponteiro global C (saida)

void set_number_threads(int n) {
    if (n > 0)
        num_threads = n; // define qtas threads usar
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

    size_t block_size = 8; // bloco 8x8 (melhor cache)
    for (size_t i = 0; i < matrix->height; i += block_size) {
        for (size_t j = 0; j < matrix->width; j += block_size) {
            for (size_t bi = 0; bi < block_size && i + bi < matrix->height; bi++) {
                for (size_t bj = 0; bj < block_size && j + bj < matrix->width; bj++) {
                    // copia elemento transposto
                    transposed->rows[(j + bj) * transposed->width + (i + bi)] =
                        matrix->rows[(i + bi) * matrix->width + (j + bj)];
                }
            }
        }
    }
    return transposed;
}

// executa a multiplicação escalar em parte da matriz atribuída a thread
void* scalar_mult_thread(void *arg) { 
    long thread_id = (long)arg;  
    Matrix *m = global_matrix_scalar;
    float scalar = global_scalar;
    size_t width = m->width;

    __m256 scalar_vec = _mm256_set1_ps(scalar); // escalar replicado em AVX

    // cada thread processa linhas intercaladas
    for (size_t i = thread_id; i < m->height; i += num_threads) {
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


int scalar_matrix_mult(float scalar_value, Matrix *matrix) {
     if (!matrix || !matrix->rows)
        return 0;

    // define vars globais por threads
    global_scalar = scalar_value;
    global_matrix_scalar = matrix;

    pthread_t threads[num_threads];

    // cria threads e esperando acabar
    for (long t = 0; t < num_threads; t++)
        pthread_create(&threads[t], NULL, scalar_mult_thread, (void*)t);

    for (int t = 0; t < num_threads; t++)
        pthread_join(threads[t], NULL);

    return 1;
}

// executa a multiplicação de matrizes nas linhas atribuídas a thread.
void* matrix_mult_thread(void *arg) {
    long thread_id = (long)arg;  // Identificador da thread

    // Matrizes globais compartilhadas
    Matrix *A = thread_matrixA;
    Matrix *B_T = thread_matrixB_transposed;
    Matrix *C = thread_matrixC;

    size_t A_height = A->height;
    size_t A_width  = A->width;
    size_t B_width  = B_T->height; // B transposta: linhas = colunas originais de B

    // Cada thread processará linhas intercaladas:
    // thread_id, thread_id + num_threads, thread_id + 2*num_threads, ...
    for (size_t i = thread_id; i < A_height; i += num_threads) {
        for (size_t j = 0; j < B_width; j++) {
            __m256 sum_vec = _mm256_setzero_ps(); // acumulador vetorial AVX
            size_t k = 0;

            // Multiplicação vetorial AVX + FMA (8 floats por vez)
            for (; k + 8 <= A_width; k += 8) {
                __m256 a = _mm256_load_ps(&A->rows[i * A_width + k]);
                __m256 b = _mm256_load_ps(&B_T->rows[j * B_T->width + k]);
                sum_vec = _mm256_fmadd_ps(a, b, sum_vec); // sum_vec += a*b
            }

            // Reduz vetor AVX para float escalar
            __m128 low  = _mm256_castps256_ps128(sum_vec);
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum  = _mm_add_ps(low, high);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            float total = _mm_cvtss_f32(sum);

            // Processa o restante (caso não múltiplo de 8)
            for (; k < A_width; k++)
                total += A->rows[i * A_width + k] * B_T->rows[j * B_T->width + k];

            // Escreve o resultado final na matriz C
            C->rows[i * C->width + j] = total;
        }
    }

    pthread_exit(NULL);
}


int matrix_matrix_mult(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC) {
   if (!matrixA || !matrixB || !matrixC)
        return 0;
    if (matrixA->width != matrixB->height)
        return 0;

    Matrix *B_transposed = getMatrixTransposed(matrixB);
    if (!B_transposed)
        return 0;

    thread_matrixA = matrixA;
    thread_matrixB_transposed = B_transposed;
    thread_matrixC = matrixC;


    pthread_t threads[num_threads];

    for (long t = 0; t < num_threads; t++)
        pthread_create(&threads[t], NULL, matrix_mult_thread, (void*)t);

    for (int t = 0; t < num_threads; t++)
        pthread_join(threads[t], NULL);

    free(B_transposed->rows);
    free(B_transposed);

    return 1;
}
