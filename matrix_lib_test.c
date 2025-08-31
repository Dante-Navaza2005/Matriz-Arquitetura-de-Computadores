#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h" 
#include "timer.h"

int getMatrixFromFile(char* path, Matrix* m){
    FILE* f = fopen(path, "rb");

    if(!f){
        printf("Erro de leitura");
        exit(1);
    }

     // Lê todos os elementos de uma vez
    size_t total = m->height * m->width;
    size_t read = fread(m->rows, sizeof(float), total, f);
    fclose(f);

    if (read != m->height * m->width) {
        fprintf(stderr, "Erro: número incorreto de elementos lidos de %s\n", path);
        return 0;
    }


    return 1;
}

int saveMatrix(char* path, Matrix *m){
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


int main(int argc, char *argv[]) {
    if (argc != 10) {
        printf("Uso: ./matrix_lib_test <escalar> <hA> <wA> <hB> <wB> <arquivoA> <arquivoB> <arquivoA_result> <arquivoC>\n");
        return 1;
    }

    struct timeval overall_t1, overall_t2, t_start, t_stop;
    gettimeofday(&overall_t1, NULL);

    // Lendo os argumentos da linha de comando
    float num_esc = atof(argv[1]);
    unsigned long int heightA = atol(argv[2]);
    unsigned long int widthA = atol(argv[3]);
    unsigned long int heightB = atol(argv[4]);
    unsigned long int widthB = atol(argv[5]);

    char* fileA = argv[6];
    char* fileB = argv[7];
    char* fileA_r = argv[8];
    char* fileC = argv[9];

    // Verificar se as matrizes A e B podem ser multiplicadas 
    if (widthA != heightB) { // witdhA é a quantidade de colunas de A e heightB é a quantidades de linhas de B
        printf("O num de colunas de A (%lu) é diferente do num de linhas de B (%lu).\n", widthA, heightB);
        return 1;
    }

    Matrix mA = {heightA, widthA, malloc(sizeof(float) * heightA * widthA)};
    Matrix mB = {heightB, widthB, malloc(sizeof(float) * heightB * widthB)};
    Matrix mC = {heightA, widthB, malloc(sizeof(float) * heightA * widthB)};

    // Inicializar a matriz C com zeros apenas
    for (int i = 0; i < heightA * widthB; i++) {
        mC.rows[i] = 0.0f;
    }

    // Populando matriz A e B dos arquivos binários
    if (!getMatrixFromFile(fileA, &mA)){
        return 1;
    } 

    printf("\nMatriz A:\n");
    for (unsigned long int linha = 0; linha < mA.height; linha++) {
        for (unsigned long int coluna = 0; coluna < mA.width; coluna++) {
            printf("%.2f ", mA.rows[linha * mA.width + coluna]);
        }
        printf("\n");
    }

    if (!getMatrixFromFile(fileB, &mB)){
        return 1;
    } 

    printf("\nMatriz B:\n");
    for (unsigned long int linha = 0; linha < mB.height; linha++) {
        for (unsigned long int coluna = 0; coluna < mB.width; coluna++) {
            printf("%.2f ", mB.rows[linha * mB.width + coluna]);
        }
        printf("\n");
    }

    // Medindo scalarMatrixMult
    gettimeofday(&t_start, NULL);
    if (!scalarMatrixMult(num_esc, &mA)) {
        printf("Erro ao calcular a multiplicação escalar de A\n");
        return 1;
    }
    gettimeofday(&t_stop, NULL);
    printf("\nTempo da scalarMatrixMult: %.3f ms\n", timedifference_msec(t_start, t_stop));

    if (!saveMatrix(fileA_r, &mA)) return 1;

    printf("\nMatriz A dps da multiplicacao escalar por %.2f:\n", num_esc);
    for (unsigned long int linha = 0; linha < mA.height; linha++) {
        for (unsigned long int coluna = 0; coluna < mA.width; coluna++) {
            printf("%.2f ", mA.rows[linha * mA.width + coluna]);
        }
        printf("\n");
    }

    // Medindo matrixMatrixMult
    gettimeofday(&t_start, NULL);
    if (!matrixMatrixMult(&mA, &mB, &mC)) {
        printf("Erro ao multiplicar as matrizes A e B\n");
        return 1;
    }
    gettimeofday(&t_stop, NULL);
    printf("\nTempo da matrixMatrixMult: %.3f ms\n", timedifference_msec(t_start, t_stop));

    // Salvando a matriz C
    if (!saveMatrix(fileC, &mC)) return 1;

    printf("\nMatriz C dps de multiplicar A e B:\n");
    for (unsigned long int linha = 0; linha < mC.height; linha++) {
        for (unsigned long int coluna = 0; coluna < mC.width; coluna++) {
            printf("%.2f ", mC.rows[linha * mC.width + coluna]);
        }
        printf("\n");
    }

    free(mA.rows);
    free(mB.rows);
    free(mC.rows);

    gettimeofday(&overall_t2, NULL);  // fim do programa
    printf("\nTempo total do programa: %.3f ms\n", timedifference_msec(overall_t1, overall_t2));


    return 0;
}