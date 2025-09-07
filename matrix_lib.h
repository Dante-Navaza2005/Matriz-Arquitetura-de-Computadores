#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct matrix Matrix;
struct matrix {
    unsigned long int height; 
    unsigned long int width;  
    float *rows;              
};

int scalarMatrixMult(float scalar_value, struct matrix *matrix);

int matrixMatrixMult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);

// Função de timer movida de timer.h
float timedifference_msec(struct timeval t0, struct timeval t1);
