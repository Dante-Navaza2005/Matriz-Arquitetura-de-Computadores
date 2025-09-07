#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"

int scalarMatrixMult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0; 
    }

    for (unsigned long int i = 0; i < matrix->height * matrix->width; i++) {
        matrix->rows[i] *= scalar_value;
    }

    return 1;
}
int matrixMatrixMult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC) {
    if (matrixA == NULL || matrixB == NULL || matrixC == NULL || matrixA->width != matrixB->height || matrixC->height != matrixA->height || matrixC->width != matrixB->width) {
        return 0; 
    }

    // Estratégia: varrer B por colunas e A por linhas, com
    // incremento de ponteiros para minimizar aritmética de índices.
    unsigned long int m = matrixA->height;   // linhas de A e C
    unsigned long int n = matrixB->width;    // colunas de B e C
    unsigned long int p = matrixA->width;    // colunas de A = linhas de B

    for (unsigned long int j = 0; j < n; j++) {
        // Ponteiro para o início da coluna j de B
        float *b_col = matrixB->rows + j; // avança por passos de width (n)

        for (unsigned long int i = 0; i < m; i++) {
            float sum = 0.0f;

            // Ponteiro para o início da linha i de A
            float *a_row = matrixA->rows + i * p;
            // Ponteiro para o elemento (0,j) de B (início da coluna j)
            float *b_ptr = b_col;

            // Acumula produto escalar da linha i de A com a coluna j de B
            for (unsigned long int k = 0; k < p; k++) {
                sum += a_row[k] * (*b_ptr);
                b_ptr += n; // desce uma linha na mesma coluna de B
            }

            matrixC->rows[i * n + j] = sum;
        }
    }

    return 1;
}

/*
* float timedifference_msec(struct timeval t0, struct timeval t1)
*
* Recebe uma marca de tempo t0 e outra marca de tempo t1 (ambas do
* tipo struct timeval) e retorna a diferença de tempo (delta) entre
* t1 e t0 em milissegundos (tipo float).
*/
float timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}
