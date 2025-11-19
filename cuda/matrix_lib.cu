#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

static int global_threads_per_block = 256;
static int global_max_blocks = 65535;   // <- maior limite permitido, não foge do enunciado

// Define grid e block dentro dos limites da arquitetura CUDA
int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    const int max_tpb = 1024;
    const int max_blk = 65535;

    if (threads_per_block > max_tpb || max_blocks_per_grid > max_blk) {
        global_threads_per_block = 256;
        global_max_blocks = 4096;
        return 0;
    }

    global_threads_per_block = threads_per_block;
    global_max_blocks = max_blocks_per_grid;
    return 1;
}


// kernel escalar
__global__
void scalar_kernel(float scalar, float *d_rows, unsigned long total) {
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long stride = blockDim.x * gridDim.x;

    for (unsigned long i = idx; i < total; i += stride) {
        d_rows[i] *= scalar;
    }
}


// kernel matmul completo (FULL_ALLOC)
__global__
void matmul_kernel(const float *A,
                   const float *B,
                   float *C,
                   unsigned long height,
                   unsigned long width_a,
                   unsigned long width_b)
{
    unsigned long total = height * width_b;
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long stride = blockDim.x * gridDim.x;

    for (unsigned long pos = idx; pos < total; pos += stride) {

        unsigned long i = pos / width_b;
        unsigned long j = pos % width_b;

        float sum = 0.0f;

        for (unsigned long k = 0; k < width_a; k++) {
            sum += A[i * width_a + k] * B[k * width_b + j];
        }

        C[pos] = sum;
    }
}


// kernel do modo partial alloc (linha de A × matriz B)
__global__
void line_kernel(const float *a_line,
                 const float *b_full,
                 float *c_line,
                 unsigned long width_a,
                 unsigned long width_b)
{
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width_b) {
        float sum = 0;

        for (unsigned long k = 0; k < width_a; k++) {
            sum += a_line[k] * b_full[k * width_b + idx];
        }

        c_line[idx] = sum;
    }
}


// escalar full e partial
int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (!matrix || !matrix->h_rows) {
        return 0;
    }

    unsigned long total = matrix->height * matrix->width;

    // full alloc
    if (matrix->alloc_mode == FULL_ALLOC) {

        int block_size = global_threads_per_block;
        int num_blocks = (total + block_size - 1) / block_size;

        if (num_blocks > global_max_blocks)
            num_blocks = global_max_blocks;

        scalar_kernel<<<num_blocks, block_size>>>(scalar_value,
                                                  matrix->d_rows,
                                                  total);

        cudaDeviceSynchronize();

        cudaMemcpy(matrix->h_rows,
                   matrix->d_rows,
                   total * sizeof(float),
                   cudaMemcpyDeviceToHost);

        return 1;
    }

    // partial alloc
    unsigned long width = matrix->width;
    unsigned long bytes = width * sizeof(float);

    float *d_line = matrix->d_rows;

    for (unsigned long i = 0; i < matrix->height; i++) {

        cudaMemcpy(d_line,
                   &matrix->h_rows[i * width],
                   bytes,
                   cudaMemcpyHostToDevice);

        int block_size = global_threads_per_block;
        int num_blocks = (width + block_size - 1) / block_size;

        if (num_blocks > global_max_blocks)
            num_blocks = global_max_blocks;

        scalar_kernel<<<num_blocks, block_size>>>(scalar_value,
                                                  d_line,
                                                  width);

        cudaMemcpy(&matrix->h_rows[i * width],
                   d_line,
                   bytes,
                   cudaMemcpyDeviceToHost);
    }

    return 1;
}


// matmul - full e partial
int matrix_matrix_mult(struct matrix *A,
                       struct matrix *B,
                       struct matrix *C)
{
    if (!A || !B || !C)
        return 0;

    if (A->width != B->height)
        return 0;

    unsigned long height = A->height;
    unsigned long width_a = A->width;
    unsigned long width_b = B->width;

    unsigned long total_c = height * width_b;


    // full alloc
    if (A->alloc_mode == FULL_ALLOC &&
        B->alloc_mode == FULL_ALLOC &&
        C->alloc_mode == FULL_ALLOC)
    {
        int block_size = global_threads_per_block;
        int num_blocks = (total_c + block_size - 1) / block_size;

        if (num_blocks > global_max_blocks)
            num_blocks = global_max_blocks;

        matmul_kernel<<<num_blocks, block_size>>>(A->d_rows,
                                                  B->d_rows,
                                                  C->d_rows,
                                                  height,
                                                  width_a,
                                                  width_b);

        cudaDeviceSynchronize();

        cudaMemcpy(C->h_rows,
                   C->d_rows,
                   total_c * sizeof(float),
                   cudaMemcpyDeviceToHost);

        return 1;
    }

    // partial alloc  (linha de A × B completa)
    unsigned long bytes_a = width_a * sizeof(float);
    unsigned long bytes_c = width_b * sizeof(float);

    float *d_line_a = A->d_rows;
    float *d_line_c = C->d_rows;
    float *d_full_b = B->d_rows;

    for (unsigned long i = 0; i < height; i++) {

        // enviar uma linha de A
        cudaMemcpy(d_line_a,
                   &A->h_rows[i * width_a],
                   bytes_a,
                   cudaMemcpyHostToDevice);

        int block_size = global_threads_per_block;
        int num_blocks = (width_b + block_size - 1) / block_size;

        if (num_blocks > global_max_blocks)
            num_blocks = global_max_blocks;

        // cálculo de uma linha de C
        line_kernel<<<num_blocks, block_size>>>(d_line_a,
                                                d_full_b,
                                                d_line_c,
                                                width_a,
                                                width_b);

        cudaMemcpy(&C->h_rows[i * width_b],
                   d_line_c,
                   bytes_c,
                   cudaMemcpyDeviceToHost);
    }

    return 1;
}
