#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"


static int GLOBAL_THREADS_PER_BLOCK = 256;
static int GLOBAL_MAX_BLOCKS = 4096;

int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    const int MAX_TPB = 1024;
    const int MAX_BLK = 65535;

    if (threads_per_block > MAX_TPB || max_blocks_per_grid > MAX_BLK) {
        GLOBAL_THREADS_PER_BLOCK = 256;
        GLOBAL_MAX_BLOCKS = 4096;
        return 0;
    }

    GLOBAL_THREADS_PER_BLOCK = threads_per_block;
    GLOBAL_MAX_BLOCKS = max_blocks_per_grid;
    return 1;
}


__global__
void scalar_kernel(float scalar, float *d_rows, unsigned long int total) {
    unsigned long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int stride = blockDim.x * gridDim.x;

    for (unsigned long int i = idx; i < total; i += stride) {
        d_rows[i] *= scalar;
    }
}


__global__
void matmul_kernel(const float *A,
                   const float *B,
                   float *C,
                   unsigned long height,
                   unsigned long widthA,
                   unsigned long widthB)
{
    unsigned long total = height * widthB;
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long stride = blockDim.x * gridDim.x;

    for (unsigned long pos = idx; pos < total; pos += stride) {
        unsigned long i = pos / widthB;
        unsigned long j = pos % widthB;

        float sum = 0.0f;
        for (unsigned long k = 0; k < widthA; k++) {
            sum += A[i * widthA + k] * B[k * widthB + j];
        }

        C[pos] = sum;
    }
}


__global__
void line_kernel(const float *Aline,
                 const float *Bfull,
                 float *Cline,
                 unsigned long widthA,
                 unsigned long widthB)
{
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < widthB) {
        float sum = 0;
        for (unsigned long k = 0; k < widthA; k++) {
            sum += Aline[k] * Bfull[k * widthB + idx];
        }
        Cline[idx] = sum;
    }
}


int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (!matrix || !matrix->h_rows)
        return 0;

    unsigned long total = matrix->height * matrix->width;

    if (matrix->alloc_mode == FULL_ALLOC) {
        int blockSize = GLOBAL_THREADS_PER_BLOCK;
        int numBlocks = (total + blockSize - 1) / blockSize;
        if (numBlocks > GLOBAL_MAX_BLOCKS) numBlocks = GLOBAL_MAX_BLOCKS;

        scalar_kernel<<<numBlocks, blockSize>>>(scalar_value,
                                                matrix->d_rows,
                                                total);

        cudaDeviceSynchronize();

        cudaMemcpy(matrix->h_rows,
                   matrix->d_rows,
                   total * sizeof(float),
                   cudaMemcpyDeviceToHost);

        return 1;
    }
    else {
        unsigned long w = matrix->width;
        unsigned long bytes = w * sizeof(float);

        float *d_line = matrix->d_rows;

        for (unsigned long i = 0; i < matrix->height; i++) {

            cudaMemcpy(d_line,
                       &matrix->h_rows[i * w],
                       bytes,
                       cudaMemcpyHostToDevice);

            int blockSize = GLOBAL_THREADS_PER_BLOCK;
            int numBlocks = (w + blockSize - 1) / blockSize;
            if (numBlocks > GLOBAL_MAX_BLOCKS) numBlocks = GLOBAL_MAX_BLOCKS;

            scalar_kernel<<<numBlocks, blockSize>>>(scalar_value,
                                                    d_line,
                                                    w);
            cudaDeviceSynchronize();

            cudaMemcpy(&matrix->h_rows[i * w],
                       d_line,
                       bytes,
                       cudaMemcpyDeviceToHost);
        }

        return 1;
    }
}


int matrix_matrix_mult(struct matrix *A,
                       struct matrix *B,
                       struct matrix *C)
{
    if (!A || !B || !C) return 0;
    if (A->width != B->height) return 0;

    unsigned long H = A->height;
    unsigned long WA = A->width;
    unsigned long WB = B->width;

    unsigned long totalC = H * WB;

    if (A->alloc_mode == FULL_ALLOC &&
        B->alloc_mode == FULL_ALLOC &&
        C->alloc_mode == FULL_ALLOC)
    {
        int blockSize = GLOBAL_THREADS_PER_BLOCK;
        int numBlocks = (totalC + blockSize - 1) / blockSize;
        if (numBlocks > GLOBAL_MAX_BLOCKS) numBlocks = GLOBAL_MAX_BLOCKS;

        matmul_kernel<<<numBlocks, blockSize>>>(A->d_rows,
                                                B->d_rows,
                                                C->d_rows,
                                                H, WA, WB);

        cudaDeviceSynchronize();

        cudaMemcpy(C->h_rows,
                   C->d_rows,
                   totalC * sizeof(float),
                   cudaMemcpyDeviceToHost);

        return 1;
    }
    else {

        unsigned long wA = WA;
        unsigned long wB = WB;
        unsigned long bytesA = wA * sizeof(float);
        unsigned long bytesC = wB * sizeof(float);

        float *d_lineA = A->d_rows; 
        float *d_lineC = C->d_rows; 
        float *d_fullB = B->d_rows; 

        for (unsigned long i = 0; i < H; i++) {

            cudaMemcpy(d_lineA,
                       &A->h_rows[i * wA],
                       bytesA,
                       cudaMemcpyHostToDevice);

            int blockSize = GLOBAL_THREADS_PER_BLOCK;
            int numBlocks = (wB + blockSize - 1) / blockSize;
            if (numBlocks > GLOBAL_MAX_BLOCKS) numBlocks = GLOBAL_MAX_BLOCKS;

            line_kernel<<<numBlocks, blockSize>>>(d_lineA,
                                                  d_fullB,
                                                  d_lineC,
                                                  wA,
                                                  wB);

            cudaDeviceSynchronize();

            cudaMemcpy(&C->h_rows[i * wB],
                       d_lineC,
                       bytesC,
                       cudaMemcpyDeviceToHost);
        }

        return 1;
    }
}
