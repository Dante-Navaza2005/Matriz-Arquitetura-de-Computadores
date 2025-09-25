/*
Dante Honorato Navaza 2321406
Maria Laura Soares 2320467
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>  
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

// Lê matriz de arquivo binário em blocos de 8 floats usando AVX
int getMatrixFromFile(const char* path, Matrix* m) {
    if (!m || !m->rows) return 0;

    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Erro de leitura do arquivo %s\n", path);
        return 0;
    }

    size_t total = m->height * m->width;

    // Verifica alinhamento
    int aligned = ((uintptr_t)m->rows % 32) == 0;

    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        float auxiliar[8]; // buffer temporário para fread
        if (fread(auxiliar, sizeof(float), 8, f) != 8) {
            printf("Erro ao ler matriz do arquivo %s\n", path);
            fclose(f);
            return 0;
        }
        __m256 vec = _mm256_loadu_ps(auxiliar);  // usa loadu_ps, pq é seguro mesmo se não estiver alinhado
        if (aligned) { // verifica alinhamento
            _mm256_store_ps(&m->rows[i], vec);
        } else {
            _mm256_storeu_ps(&m->rows[i], vec);
        }
    }

    // Elementos restantes se tiver (no nosso caso não ter pq é multiplo de 8)
    for (; i < total; i++) {
        if (fread(&m->rows[i], sizeof(float), 1, f) != 1) {
            printf("Erro ao ler matriz do arquivo %s\n", path);
            fclose(f);
            return 0;
        }
    }

    fclose(f);
    return 1;
}

// Salva matriz em arquivo binário em blocos de 8 floats usando AVX
int saveMatrix(const char* path, Matrix* m) {
    if (!m || !m->rows) return 0;

    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("Erro ao abrir arquivo %s para escrita\n", path);
        return 0;
    }

    size_t total = m->height * m->width;
    int aligned = ((uintptr_t)m->rows % 32) == 0;

    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 vec;
        if (aligned) {
            vec = _mm256_load_ps(&m->rows[i]);
        } else {
            vec = _mm256_loadu_ps(&m->rows[i]);
        }

        float auxiliar[8];
        _mm256_storeu_ps(auxiliar, vec);  // storeu é seguro para gravar em buffer temporário

        if (fwrite(auxiliar, sizeof(float), 8, f) != 8) {
            printf("Erro ao escrever matriz no arquivo %s\n", path);
            fclose(f);
            return 0;
        }
    }

    // Elementos restantes
    for (; i < total; i++) {
        if (fwrite(&m->rows[i], sizeof(float), 1, f) != 1) {
            printf("Erro ao escrever matriz no arquivo %s\n", path);
            fclose(f);
            return 0;
        }
    }

    fclose(f);
    return 1;
}


int initializeWithZeros(Matrix *m){
    size_t total = m->height * m->width;
    if (total % 8 != 0) {
        return 0; 
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