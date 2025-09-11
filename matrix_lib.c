/*
Dante Honorato Navaza 2321406
Maria Laura Soares 2320467
*/

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

    unsigned long int matrixA_height = matrixA->height;  
    unsigned long int matrixB_width = matrixB->width;   
    unsigned long int matrixA_width = matrixA->width;   

    for (unsigned long int col = 0; col < matrixB_width; col++) { 
        float *b_col = matrixB->rows + col;  // pega o início da coluna

        for (unsigned long int row = 0; row < matrixA_height; row++) {
            float sum = 0.0f;

            float *a_row = matrixA->rows + row * matrixA_width;
            float *b_ptr = b_col;

            for (unsigned long int k = 0; k < matrixA_width; k++) {
                sum += a_row[k] * (*b_ptr);
                b_ptr += matrixB_width;  // desce uma linha na coluna de B
            }

            matrixC->rows[row * matrixB_width + col] = sum;
        }
    }

    return 1;
}
