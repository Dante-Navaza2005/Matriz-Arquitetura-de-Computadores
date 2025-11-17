#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "timer.h"
#include "matrix_lib.h"



void load_float_file(const char *path, float *buffer, unsigned long count) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Erro lendo %s\n", path); exit(1); }
    fread(buffer, sizeof(float), count, f);
    fclose(f);
}

void save_float_file(const char *path, float *buffer, unsigned long count) {
    FILE *f = fopen(path, "wb");
    if (!f) { printf("Erro escrevendo %s\n", path); exit(1); }
    fwrite(buffer, sizeof(float), count, f);
    fclose(f);
}

void print_matrix_preview(const char *name, float *m,
                          unsigned long H, unsigned long W)
{
    printf("Matriz %s (até 256 elementos):\n", name);

    unsigned long total = H * W;
    unsigned long limit = (total < 256 ? total : 256);

    for (unsigned long i = 0; i < limit; i++)
        printf("%.2f ", m[i]);

    printf("\n\n");
}


int main(int argc, char *argv[]) {

    if (argc != 13) {
        printf("Uso:\n");
        printf("matrix_lib_test scalar Ah Aw Bh Bw TPB MaxBlocks MaxMiB A.dat B.dat out1.dat out2.dat\n");
        return 1;
    }

    float scalar = atof(argv[1]);
    unsigned long Ah = atol(argv[2]);
    unsigned long Aw = atol(argv[3]);
    unsigned long Bh = atol(argv[4]);
    unsigned long Bw = atol(argv[5]);
    int TPB = atoi(argv[6]);
    int MaxBlocks = atoi(argv[7]);
    unsigned long MaxMiB = atol(argv[8]);

    const char *fileA = argv[9];
    const char *fileB = argv[10];
    const char *fileR1 = argv[11];
    const char *fileR2 = argv[12];

    if (Aw != Bh) {
        printf("Dimensoes incompativeis A.width != B.height\n");
        return 1;
    }


    unsigned long sizeA = Ah * Aw;
    unsigned long sizeB = Bh * Bw;
    unsigned long sizeC = Ah * Bw;

    struct matrix A, B, C;

    A.height = Ah; A.width = Aw;
    A.h_rows = (float*) malloc(sizeA*sizeof(float));

    B.height = Bh; B.width = Bw;
    B.h_rows = (float*) malloc(sizeB*sizeof(float));

    C.height = Ah; C.width = Bw;
    C.h_rows = (float*) malloc(sizeC*sizeof(float));

    load_float_file(fileA, A.h_rows, sizeA);
    load_float_file(fileB, B.h_rows, sizeB);


    unsigned long totalBytes =
        (sizeA + sizeB + sizeC) * sizeof(float);

    unsigned long maxBytes = MaxMiB * 1024ULL * 1024ULL;

    int full_ok = (totalBytes <= maxBytes);

    if (full_ok) {
        cudaMalloc(&A.d_rows, sizeA*sizeof(float));
        cudaMalloc(&B.d_rows, sizeB*sizeof(float));
        cudaMalloc(&C.d_rows, sizeC*sizeof(float));

        A.alloc_mode = FULL_ALLOC;
        B.alloc_mode = FULL_ALLOC;
        C.alloc_mode = FULL_ALLOC;

        cudaMemcpy(A.d_rows, A.h_rows, sizeA*sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(B.d_rows, B.h_rows, sizeB*sizeof(float),
                   cudaMemcpyHostToDevice);

    } else {

        /* PARTIAL_ALLOC: B inteiro + 1 linha de A + 1 linha de C */

        unsigned long bytesAline = Aw*sizeof(float);
        unsigned long bytesCline = Bw*sizeof(float);

        unsigned long need =
            sizeB*sizeof(float) + bytesAline + bytesCline;

        if (need > maxBytes) {
            printf("Nao cabe nem partial alloc.\n");
            return 1;
        }

        cudaMalloc(&B.d_rows, sizeB*sizeof(float));
        cudaMalloc(&A.d_rows, bytesAline);
        cudaMalloc(&C.d_rows, bytesCline);

        A.alloc_mode = PARTIAL_ALLOC;
        B.alloc_mode = FULL_ALLOC;
        C.alloc_mode = PARTIAL_ALLOC;

        cudaMemcpy(B.d_rows, B.h_rows, sizeB*sizeof(float),
                   cudaMemcpyHostToDevice);
    }


    set_grid_size(TPB, MaxBlocks);


    struct timeval t0, t1, t2;

    gettimeofday(&t0, NULL);

    gettimeofday(&t1, NULL);
    scalar_matrix_mult(scalar, &A);
    gettimeofday(&t2, NULL);

    double scalar_ms = timedifference_msec(t1, t2);

    save_float_file(fileR1, A.h_rows, sizeA);

    gettimeofday(&t1, NULL);
    matrix_matrix_mult(&A, &B, &C);
    gettimeofday(&t2, NULL);

    double matmul_ms = timedifference_msec(t1, t2);

    save_float_file(fileR2, C.h_rows, sizeC);

    double total_ms = timedifference_msec(t0, t2);


    print_matrix_preview("A", A.h_rows, Ah, Aw);
    print_matrix_preview("B", B.h_rows, Bh, Bw);
    print_matrix_preview("C", C.h_rows, Ah, Bw);

    printf("Tempo scalar = %.4f ms\n", scalar_ms);
    printf("Tempo matmul = %.4f ms\n", matmul_ms);
    printf("Tempo total  = %.4f ms\n", total_ms);

    printf("\n------ PROCESSADOR (lscpu) ------\n");
    system("lscpu");

    return 0;
}
