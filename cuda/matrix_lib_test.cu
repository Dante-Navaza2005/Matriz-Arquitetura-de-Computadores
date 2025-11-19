#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "timer.h"
#include "matrix_lib.h"


// Carrega um arquivo binário contendo floats
void load_float_file(const char *path, float *buffer, unsigned long count) {
    FILE *arquivo = fopen(path, "rb");

    if (!arquivo) {
        printf("Erro lendo %s\n", path);
        exit(1);
    }

    fread(buffer, sizeof(float), count, arquivo);
    fclose(arquivo);
}

// Salva floats em arquivo binário
void save_float_file(const char *path, float *buffer, unsigned long count) {
    FILE *arquivo = fopen(path, "wb");

    if (!arquivo) {
        printf("Erro escrevendo %s\n", path);
        exit(1);
    }

    fwrite(buffer, sizeof(float), count, arquivo);
    fclose(arquivo);
}

// Imprime até 256 elementos da matriz
void print_matrix_preview(const char *name, float *m,
                          unsigned long height, unsigned long width)
{
    printf("Matriz %s (até 256 elementos):\n", name);

    unsigned long total = height * width;
    unsigned long limite = (total < 256 ? total : 256);

    for (unsigned long i = 0; i < limite; i++) {
        printf("%.2f ", m[i]);
    }

    printf("\n\n");
}


int main(int argc, char *argv[]) {

    // Confere quantidade de argumentos
    if (argc != 13) {
        printf("Uso:\n");
        printf("matrix_lib_test scalar Ah Aw Bh Bw TPB MaxBlocks MaxMiB A.dat B.dat out1.dat out2.dat\n");
        return 1;
    }

    // Lê argumentos do usuário
    float scalar = atof(argv[1]);
    unsigned long altura_a = atol(argv[2]);
    unsigned long largura_a = atol(argv[3]);
    unsigned long altura_b = atol(argv[4]);
    unsigned long largura_b = atol(argv[5]);
    int threads_por_bloco = atoi(argv[6]);
    int max_blocos = atoi(argv[7]);
    unsigned long max_mib = atol(argv[8]);

    const char *arquivo_a = argv[9];
    const char *arquivo_b = argv[10];
    const char *arquivo_saida1 = argv[11];
    const char *arquivo_saida2 = argv[12];

    // Matrizes compatíveis?
    if (largura_a != altura_b) {
        printf("Dimensoes incompativeis A.width != B.height\n");
        return 1;
    }

    // Quantidade de floats em cada matriz
    unsigned long tamanho_a = altura_a * largura_a;
    unsigned long tamanho_b = altura_b * largura_b;
    unsigned long tamanho_c = altura_a * largura_b;

    // Structs das matrizes
    struct matrix A, B, C;

    // Alocação na RAM
    A.height = altura_a;
    A.width = largura_a;
    A.h_rows = (float*) malloc(tamanho_a * sizeof(float));

    B.height = altura_b;
    B.width = largura_b;
    B.h_rows = (float*) malloc(tamanho_b * sizeof(float));

    C.height = altura_a;
    C.width = largura_b;
    C.h_rows = (float*) malloc(tamanho_c * sizeof(float));

    // Carrega arquivos .dat
    load_float_file(arquivo_a, A.h_rows, tamanho_a);
    load_float_file(arquivo_b, B.h_rows, tamanho_b);

    // Calcula bytes totais necessários
    unsigned long total_bytes =
        (tamanho_a + tamanho_b + tamanho_c) * sizeof(float);

    unsigned long max_bytes = max_mib * 1024ULL * 1024ULL;

    int full_ok = (total_bytes <= max_bytes);

    // Tenta FULL_ALLOC
    if (full_ok) {
        cudaMalloc(&A.d_rows, tamanho_a * sizeof(float));
        cudaMalloc(&B.d_rows, tamanho_b * sizeof(float));
        cudaMalloc(&C.d_rows, tamanho_c * sizeof(float));

        A.alloc_mode = FULL_ALLOC;
        B.alloc_mode = FULL_ALLOC;
        C.alloc_mode = FULL_ALLOC;

        // Copia matrizes A e B para a GPU
        cudaMemcpy(A.d_rows, A.h_rows, tamanho_a * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(B.d_rows, B.h_rows, tamanho_b * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
    else {
        // PARTIAL_ALLOC: B inteira + 1 linha de A + 1 linha de C
        unsigned long bytes_linha_a = largura_a * sizeof(float);
        unsigned long bytes_linha_c = largura_b * sizeof(float);

        unsigned long necessario =
            tamanho_b * sizeof(float) + bytes_linha_a + bytes_linha_c;

        if (necessario > max_bytes) {
            printf("Nao cabe nem partial alloc.\n");
            return 1;
        }

        // Alocação parcial na GPU
        cudaMalloc(&B.d_rows, tamanho_b * sizeof(float));
        cudaMalloc(&A.d_rows, bytes_linha_a);
        cudaMalloc(&C.d_rows, bytes_linha_c);

        A.alloc_mode = PARTIAL_ALLOC;
        B.alloc_mode = FULL_ALLOC;
        C.alloc_mode = PARTIAL_ALLOC;

        // Copia toda B para a GPU
        cudaMemcpy(B.d_rows, B.h_rows, tamanho_b * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    // Configura grid e blocos
    set_grid_size(threads_por_bloco, max_blocos);

    // Variáveis de tempo
    struct timeval t0, t1, t2;

    gettimeofday(&t0, NULL);

    // Multiplicação escalar
    gettimeofday(&t1, NULL);
    scalar_matrix_mult(scalar, &A);
    gettimeofday(&t2, NULL);

    double tempo_scalar = timedifference_msec(t1, t2);

    save_float_file(arquivo_saida1, A.h_rows, tamanho_a);

    // Multiplicação de matrizes
    gettimeofday(&t1, NULL);
    matrix_matrix_mult(&A, &B, &C);
    gettimeofday(&t2, NULL);

    double tempo_matmul = timedifference_msec(t1, t2);

    save_float_file(arquivo_saida2, C.h_rows, tamanho_c);

    double tempo_total = timedifference_msec(t0, t2);

    // Imprime prévias das matrizes
    print_matrix_preview("A", A.h_rows, altura_a, largura_a);
    print_matrix_preview("B", B.h_rows, altura_b, largura_b);
    print_matrix_preview("C", C.h_rows, altura_a, largura_b);

    printf("Tempo scalar = %.4f ms\n", tempo_scalar);
    printf("Tempo matmul = %.4f ms\n", tempo_matmul);
    printf("Tempo total  = %.4f ms\n", tempo_total);

    printf("\n------ PROCESSADOR (lscpu) ------\n");
    system("lscpu");

    return 0;
}