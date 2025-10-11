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

/* Função auxiliar: imprime até 256 elementos da matriz */
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

/* Função para ler matriz de arquivo binário */
int getMatrixFromFile(const char* path, Matrix* m) {
    if (!m || !m->rows)
        return 0;
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Erro de leitura do arquivo %s\n", path);
        return 0;
    }
    size_t total = m->height * m->width;
    int aligned = ((uintptr_t)m->rows % 32) == 0;
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        float aux[8];
        if (fread(aux, sizeof(float), 8, f) != 8) {
            printf("Erro ao ler %s\n", path);
            fclose(f);
            return 0;
        }
        __m256 vec = _mm256_loadu_ps(aux);
        if (aligned)
            _mm256_store_ps(&m->rows[i], vec);
        else
            _mm256_storeu_ps(&m->rows[i], vec);
    }
    for (; i < total; i++)
        fread(&m->rows[i], sizeof(float), 1, f);
    fclose(f);
    return 1;
}

/* Função para salvar matriz em arquivo binário */
int saveMatrix(const char* path, Matrix* m) {
    if (!m || !m->rows)
        return 0;
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("Erro ao abrir %s\n", path);
        return 0;
    }
    size_t total = m->height * m->width;
    for (size_t i = 0; i < total; i++)
        fwrite(&m->rows[i], sizeof(float), 1, f);
    fclose(f);
    return 1;
}

/* Inicializa matriz com zeros */
int initializeWithZeros(Matrix *m) {
    size_t total = m->height * m->width;
    if (total % 8 != 0) return 0;
    __m256 zeros = _mm256_setzero_ps();
    for (size_t i = 0; i < total; i += 8)
        _mm256_store_ps(m->rows + i, zeros);
    return 1;
}

/* ============================= MAIN ============================= */
int main(int argc, char *argv[]) {
    if (argc != 11) {
        printf("Uso: ./matrix_lib_test <escalar> <hA> <wA> <hB> <wB> <n_threads> <arquivoA> <arquivoB> <arquivoA_result> <arquivoC>\n");
        return 1;
    }

    struct timeval start, stop, overall_t1, overall_t2;
    gettimeofday(&overall_t1, NULL);

    float num_esc = atof(argv[1]);
    unsigned long int heightA = atol(argv[2]);
    unsigned long int widthA = atol(argv[3]);
    unsigned long int heightB = atol(argv[4]);
    unsigned long int widthB = atol(argv[5]);
    int n_threads = atoi(argv[6]);
    const char* fileA = argv[7];
    const char* fileB = argv[8];
    const char* fileA_r = argv[9];
    const char* fileC = argv[10];

    if (widthA != heightB) {
        printf("O número de colunas de A (%lu) difere do número de linhas de B (%lu)\n", widthA, heightB);
        return 1;
    }

    set_number_threads(n_threads);

    Matrix matrixA = {heightA, widthA, aligned_alloc(32, heightA * widthA * sizeof(float))};
    Matrix matrixB = {heightB, widthB, aligned_alloc(32, heightB * widthB * sizeof(float))};
    Matrix matrixC = {heightA, widthB, aligned_alloc(32, heightA * widthB * sizeof(float))};

    if (!matrixA.rows || !matrixB.rows || !matrixC.rows) {
        printf("Erro de alocação de memória\n");
        exit(1);
    }

    if (!getMatrixFromFile(fileA, &matrixA)) return 1;
    if (!getMatrixFromFile(fileB, &matrixB)) return 1;
    if (!initializeWithZeros(&matrixC)) return 1;

    print_matrix_limited("Matriz A", &matrixA);
    print_matrix_limited("Matriz B", &matrixB);

    /* Multiplicação escalar */
    gettimeofday(&start, NULL);
    if (!scalarMatrixMult(num_esc, &matrixA)) {
        printf("Erro na multiplicação escalar\n");
        return 1;
    }
    gettimeofday(&stop, NULL);
    printf("\nTempo da multiplicação escalar: %.3f ms\n", timedifference_msec(start, stop));

    saveMatrix(fileA_r, &matrixA);

    /* Multiplicação de matrizes */
    gettimeofday(&start, NULL);
    if (!matrixMatrixMult(&matrixA, &matrixB, &matrixC)) {
        printf("Erro na multiplicação de matrizes\n");
        return 1;
    }
    gettimeofday(&stop, NULL);
    printf("\nTempo da multiplicação de matrizes: %.3f ms\n", timedifference_msec(start, stop));

    saveMatrix(fileC, &matrixC);

    print_matrix_limited("Matriz C final", &matrixC);

    free(matrixA.rows);
    free(matrixB.rows);
    free(matrixC.rows);

    gettimeofday(&overall_t2, NULL);
    printf("\nTempo total do programa: %.3f ms\n", timedifference_msec(overall_t1, overall_t2));

    return 0;
}
