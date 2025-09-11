/*
Dante Honorato Navaza 2321406
Maria Laura 2320467
*/

#include <stdio.h>
#include <stdlib.h>

typedef struct matrix Matrix;
struct matrix {
    unsigned long int height; 
    unsigned long int width;  
    float *rows;              
};

int scalarMatrixMult(float scalar_value, struct matrix *matrix);

int matrixMatrixMult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);
