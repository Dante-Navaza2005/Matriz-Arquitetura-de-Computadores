#include <stdio.h>
#include <stdlib.h>


struct matrix {
    unsigned long int height; 
    unsigned long int width;  
    float *rows;              
};

/*
height = número de linhas da matriz (múltiplo de 8)
width = número de colunas da matriz (múltiplo de 8)
rows = sequência de linhas da matriz (height*width elementos)
*/

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0; 
    }

    for (unsigned long int i = 0; i < matrix->height * matrix->width; i++) {
        matrix->rows[i] *= scalar_value;
    }

    return 1;
}
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC) {
    if (matrixA == NULL || matrixB == NULL || matrixC == NULL || matrixA->width != matrixB->height || matrixC->height != matrixA->height || matrixC->width != matrixB->width) {
        return 0; 
    }

    for (unsigned long int i = 0; i < matrixC->height * matrixC->width; i++) {
        matrixC->rows[i] = 0.0f;
    }

    for (unsigned long int i = 0; i < matrixA->height; i++) {
        for (unsigned long int j = 0; j < matrixB->width; j++) {
            for (unsigned long int k = 0; k < matrixA->width; k++) {
                matrixC->rows[i * matrixC->width + j] += matrixA->rows[i * matrixA->width + k] * matrixB->rows[k * matrixB->width + j];
            }
        }
    }

    return 1;
}


int main() {
    struct matrix matrizA = {8, 8, NULL};
    matrizA.rows = (float *)malloc(matrizA.height * matrizA.width * sizeof(float));
    
    for (unsigned long int linha = 0; linha < matrizA.height; linha++) {
        for (unsigned long int coluna = 0; coluna < matrizA.width; coluna++) {
            matrizA.rows[linha * matrizA.width + coluna] = linha * matrizA.width + coluna + 1.0f;
        }
    }

    struct matrix matrizB = {8, 8, NULL};
    matrizB.rows = (float *)malloc(matrizB.height * matrizB.width * sizeof(float));
    
    for (unsigned long int linha = 0; linha < matrizB.height; linha++) {
        for (unsigned long int coluna = 0; coluna < matrizB.width; coluna++) {
            matrizB.rows[linha * matrizB.width + coluna] = (linha + 1) * (coluna + 1);
        }
    }

    struct matrix matrizC = {8, 8, NULL};
    matrizC.rows = (float *)malloc(matrizC.height * matrizC.width * sizeof(float));

    printf("Matriz A:\n");
    for (unsigned long int linha = 0; linha < matrizA.height; linha++) {
        for (unsigned long int coluna = 0; coluna < matrizA.width; coluna++) {
            printf("%.2f ", matrizA.rows[linha * matrizA.width + coluna]);
        }
        printf("\n");
    }

    float valor_scalar = 2.0f;
    scalar_matrix_mult(valor_scalar, &matrizA);

    printf("\nMatriz A dps da multiplicacao escalar por %.2f:\n", valor_scalar);
    for (unsigned long int linha = 0; linha < matrizA.height; linha++) {
        for (unsigned long int coluna = 0; coluna < matrizA.width; coluna++) {
            printf("%.2f ", matrizA.rows[linha * matrizA.width + coluna]);
        }
        printf("\n");
    }

    matrix_matrix_mult(&matrizA, &matrizB, &matrizC);

    printf("\nMatriz C dps de multiplicar A e B:\n");
    for (unsigned long int linha = 0; linha < matrizC.height; linha++) {
        for (unsigned long int coluna = 0; coluna < matrizC.width; coluna++) {
            printf("%.2f ", matrizC.rows[linha * matrizC.width + coluna]);
        }
        printf("\n");
    }

    free(matrizA.rows);
    free(matrizB.rows);
    free(matrizC.rows);

    return 0;
}