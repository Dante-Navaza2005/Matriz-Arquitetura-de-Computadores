/*
Dante Honorato Navaza 2321406
Maria Laura Soares 2320467
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <sys/time.h>
#include "matrix_lib.h"
#include "timer.h"

// Função para imprimir matriz limitada
static void print_matrix_limited(const char *label, const Matrix *m) {
    unsigned long int total = m->height * m->width;
    unsigned long int limit = total > 256UL ? 256UL : total;
    printf("\n(%s %lu X %lu) até %lu elementos:\n", label, m->height, m->width, limit);
    unsigned long int printed = 0;
    for (unsigned long int row = 0; row < m->height && printed < limit; row++) {
        for (unsigned long int col = 0; col < m->width && printed < limit; col++) {
            printf("%.2f ", m->rows[row * m->width + col]);
            printed++;
        }
        printf("\n");
    }
}

// Lê matriz de arquivo diretamente para memória alinhada
int getMatrixFromFile(const char* path, Matrix* m) {
    if (!m || !m->rows) return 0;
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Erro de leitura do arquivo %s\n", path); return 0; }
    size_t total = m->height * m->width;
    if (fread(m->rows, sizeof(float), total, f) != total) {
        printf("Erro ao ler matriz do arquivo %s\n", path);
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

// Salva matriz em arquivo
int saveMatrix(const char* path, Matrix* m) {
    if (!m || !m->rows) return 0;
    FILE* f = fopen(path, "wb");
    if (!f) { printf("Erro ao abrir %s\n", path); return 0; }
    fwrite(m->rows, sizeof(float), m->height * m->width, f);
    fclose(f);
    return 1;
}

// Inicializa matriz com zeros usando AVX
void initializeWithZeros(Matrix *m) {
    size_t total = m->height * m->width;
    size_t i;
    __m256 zero_vec = _mm256_setzero_ps();
    for (i = 0; i + 8 <= total; i += 8)
        _mm256_store_ps(&m->rows[i], zero_vec);
    for (; i < total; i++)
        m->rows[i] = 0.0f;
}

int main(int argc, char *argv[]) {

    printf("\n****Info do Processador****\n");

    // Imprime nome do processador
    system("lscpu | grep -i 'model name' | awk -F: '{print \"Processador: \" $2}'");
    // Threads por core
    system("lscpu | grep -i 'Thread(s) per core' | awk -F: '{print \"Thread(s) per core: \" $2}'");
    // Cores por socket
    system("lscpu | grep -i 'Core(s) per socket' | awk -F: '{print \"Core(s) per socket: \" $2}'");
    // Arquitetura
    system("lscpu | grep -i 'Architecture' | awk -F: '{print \"Architecture: \" $2}'");

    if (argc != 11) {
        printf("Uso: ./matrix_lib_test <escalar> <hA> <wA> <hB> <wB> <n_threads> <arquivoA> <arquivoB> <arquivoA_result> <arquivoC>\n");
        return 1;
    }

    struct timeval start, stop, overall_t1, overall_t2;
    gettimeofday(&overall_t1, NULL);

    float scalar = atof(argv[1]);
    unsigned long int hA = atol(argv[2]), wA = atol(argv[3]);
    unsigned long int hB = atol(argv[4]), wB = atol(argv[5]);
    int n_threads = atoi(argv[6]);
    const char *fileA = argv[7], *fileB = argv[8], *fileA_r = argv[9], *fileC = argv[10];

    if (wA != hB) { printf("Erro: colunas de A != linhas de B\n"); return 1; }

    set_number_threads(n_threads);

    Matrix A = {hA, wA, aligned_alloc(32, hA * wA * sizeof(float))};
    Matrix B = {hB, wB, aligned_alloc(32, hB * wB * sizeof(float))};
    Matrix C = {hA, wB, aligned_alloc(32, hA * wB * sizeof(float))};
    if (!A.rows || !B.rows || !C.rows) { printf("Erro de alocação\n"); return 1; }

    if (!getMatrixFromFile(fileA, &A) || !getMatrixFromFile(fileB, &B)) return 1;
    initializeWithZeros(&C);

    print_matrix_limited("Matriz A", &A);
    print_matrix_limited("Matriz B", &B);

    // multiplicação escalar
    gettimeofday(&start, NULL);
    scalar_matrix_mult(scalar, &A);
    gettimeofday(&stop, NULL);
    printf("\nTempo multiplicação escalar: %.3f ms\n", timedifference_msec(start, stop));
    saveMatrix(fileA_r, &A);

    // multiplicação de matrizes
    gettimeofday(&start, NULL);
    matrix_matrix_mult(&A, &B, &C);
    gettimeofday(&stop, NULL);
    printf("\nTempo multiplicação de matrizes: %.3f ms\n", timedifference_msec(start, stop));
    saveMatrix(fileC, &C);

    print_matrix_limited("Matriz C final", &C);

    free(A.rows);
    free(B.rows);
    free(C.rows);

    gettimeofday(&overall_t2, NULL);
    printf("\nTempo total do programa: %.3f ms\n", timedifference_msec(overall_t1, overall_t2));

    return 0;
}
