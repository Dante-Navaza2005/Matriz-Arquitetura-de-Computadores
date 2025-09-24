/*
Dante Honorato Navaza 2321406
Maria Laura Soares 2320467
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>  // AVX intrinsics
#include "matrix_lib.h" 
#include "timer.h"

static void print_matrix_limited(const char *label, const Matrix *m) {
    unsigned long int total = m->height * m->width;
    unsigned long int limit = total > 256UL ? 256UL : total;

    printf("\n(%s %lu X %lu) até %lu elementos:\n",
           label, m->height, m->width, limit);

    unsigned long int printed = 0;
    for (unsigned long int row = 0; row < m->height && printed < limit; row++) {
        for (unsigned long int col = 0; col < m->width && printed < limit; col++) {
            printf("%.2f ", m->rows[row * m->width + col]);
            printed++;
        }
        printf("\n");
    }
}



int getMatrixFromFile(const char* path, Matrix* m){
    FILE* f = fopen(path, "rb");

    if(!f){
        printf("Erro de leitura");
        exit(1);
    }

    size_t total = m->height * m->width;
    if (total % 8 != 0) {
        return 0; // erro: não múltiplo de 8, precisa estar alinhado
    }
    float* auxiliar = (float*) aligned_alloc(32, total * sizeof(float));
    if (!auxiliar) {
        printf("Erro de alocação");
        fclose(f);
        exit(1);
    }
    if (fread(auxiliar, sizeof(float), total, f) != total) {
        printf("Erro ao ler matriz do arquivo %s", path);
        free(auxiliar);
        fclose(f);
        return 0;
    }
    fclose(f);

    // Copiar do auxiliar para m->rows com AVX (tudo alinhado e múltiplo de 8)
    for (size_t i = 0; i < total; i += 8) {
        __m256 v = _mm256_load_ps(auxiliar + i);
        _mm256_store_ps(m->rows + i, v);
    }

    free(auxiliar);
    return 1;
}

int saveMatrix(const char* path, Matrix *m){
    FILE* f = fopen(path, "wb");

    if(!f){
        printf("Erro de leitura");
        exit(1);
    }

    size_t written = fwrite(m->rows, sizeof(float), m->height * m->width, f);
    fclose(f);

    if (written != m->height * m->width) {
        fprintf(stderr, "Erro: número incorreto de elementos escritos em %s\n", path);
        return 0;
    }
    return 1;
}

int initializeWithZeros(Matrix *m){
    size_t total = m->height * m->width;
    if (total % 8 != 0) {
        return 0; // erro: não múltiplo de 8, precisa estar alinhado
    }
    __m256 zeros = _mm256_setzero_ps();
    for (size_t i = 0; i < total; i += 8) {
        _mm256_store_ps(m->rows + i, zeros);
    }
    return 1;
}


int main(int argc, char *argv[]) {
    
    if (argc != 10) {
        printf("Uso: ./matrix_lib_test <escalar> <hA> <wA> <hB> <wB> <arquivoA> <arquivoB> <arquivoA_result> <arquivoC>\n");
        return 1;
    }

    struct timeval start, stop, overall_t1, overall_t2;
    gettimeofday(&overall_t1, NULL);

    float num_esc = atof(argv[1]);
    unsigned long int heightA = atol(argv[2]);
    unsigned long int widthA = atol(argv[3]);
    unsigned long int heightB = atol(argv[4]);
    unsigned long int widthB = atol(argv[5]);

    const char* fileA = argv[6];
    const char* fileB = argv[7];
    const char* fileA_r = argv[8];
    const char* fileC = argv[9];

    if (widthA != heightB) { 
        printf("O num de colunas de A (%lu) é diferente do num de linhas de B (%lu).\n", widthA, heightB);
        return 1;
    }

    Matrix matrixA = {heightA, widthA, aligned_alloc(32, heightA * widthA * sizeof(float))};
    Matrix matrixB = {heightB, widthB, aligned_alloc(32, heightB * widthB * sizeof(float))};
    Matrix matrixC = {heightA, widthB, aligned_alloc(32, heightA * widthB * sizeof(float))};
   
    if (!matrixA.rows || !matrixB.rows || !matrixC.rows) {
        printf("Erro de alocação de memória\n");
        exit(1);
    }

    if (!getMatrixFromFile(fileA, &matrixA)){
        return 1;
    } 
    print_matrix_limited("Matriz A", &matrixA);

    if (!getMatrixFromFile(fileB, &matrixB)){
        return 1;
    } 
    print_matrix_limited("Matriz B", &matrixB);

    if(!initializeWithZeros(&matrixC)){
        return 1;
    }
    print_matrix_limited("Matriz C inicializada (zerada)", &matrixC);

    gettimeofday(&start, NULL);
    if (!scalarMatrixMult(num_esc, &matrixA)) {
        printf("Erro ao calcular a multiplicação escalar de A\n");
        return 1;
    }
    gettimeofday(&stop, NULL);
    printf("\nTempo da multiplicacao da matriz por escalar: %.3f ms\n", timedifference_msec(start, stop));

    if (!saveMatrix(fileA_r, &matrixA)) return 1;
    printf("\nMatriz A dps da multiplicacao escalar por %.2f:", num_esc);
    print_matrix_limited("", &matrixA);

    gettimeofday(&start, NULL);
    if (!matrixMatrixMult(&matrixA, &matrixB, &matrixC)) {
        printf("Erro ao multiplicar as matrizes A e B\n");
        return 1;
    }
    gettimeofday(&stop, NULL);
    printf("\nTempo da multiplicacao entre as matrizes A e B: %.3f ms\n", timedifference_msec(start, stop));

    if (!saveMatrix(fileC, &matrixC)) return 1;
    print_matrix_limited("Matriz C dps de multiplicar A e B", &matrixC);

    free(matrixA.rows);
    free(matrixB.rows);
    free(matrixC.rows);

    gettimeofday(&overall_t2, NULL);  
    printf("\nTempo total do programa: %.3f ms\n", timedifference_msec(overall_t1, overall_t2));


    return 0;
}
