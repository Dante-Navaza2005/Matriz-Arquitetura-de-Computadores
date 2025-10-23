/*
Dante Honorato Navaza 2321406
Maria Laura Soares 2320467
*/

#include <stdio.h>
#include <stdlib.h>

typedef struct matrix Matrix;
struct matrix {
    unsigned long int height; 
    unsigned long int width;  
    float *rows;              
};

void set_number_threads(int n);
Matrix* getMatrixTransposed(Matrix* matrix);
void* scalar_mult_thread(void *arg);
int scalar_matrix_mult(float scalar_value, Matrix *matrix);
void* matrix_mult_thread(void *arg);
int matrix_matrix_mult(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC);