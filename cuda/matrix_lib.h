#ifndef MATRIX_LIB_H
#define MATRIX_LIB_H

#define FULL_ALLOC 1
#define PARTIAL_ALLOC 0

struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *h_rows;   
    float *d_rows;   
    int alloc_mode;  
};

int set_grid_size(int threads_per_block, int max_blocks_per_grid);

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);

int matrix_matrix_mult(struct matrix *A,
                       struct matrix *B,
                       struct matrix *C);

#endif
