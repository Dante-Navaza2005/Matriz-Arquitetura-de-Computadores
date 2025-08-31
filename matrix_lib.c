#include <stdio.h>
#include <stdlib.h>

typedef struct matrix Matrix;
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