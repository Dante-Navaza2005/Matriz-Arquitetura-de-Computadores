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

    unsigned long int m = matrixA->height;  
    unsigned long int n = matrixB->width;   
    unsigned long int p = matrixA->width;   

    for (unsigned long int j = 0; j < n; j++) {
        float *b_col = matrixB->rows + j; 

        for (unsigned long int i = 0; i < m; i++) {
            float sum = 0.0f;

            float *a_row = matrixA->rows + i * p;
            float *b_ptr = b_col;

            for (unsigned long int k = 0; k < p; k++) {
                sum += a_row[k] * (*b_ptr);
                b_ptr += n; 
            }

            matrixC->rows[i * n + j] = sum;
        }
    }

    return 1;
}
