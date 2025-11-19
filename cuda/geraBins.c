#include <stdio.h>
#include <stdlib.h>

#define DIMENSION 2048 

int main(void) {
    FILE *arquivoA, *arquivoB;
    float value;
    long long total;
    long long i;

    arquivoA = fopen("floats_256_2.0f.dat", "wb");
    if (!arquivoA) {
        perror("Erro ao abrir floats_256_2.0f.dat");
        return 1;
    }

    arquivoB = fopen("floats_256_5.0f.dat", "wb");
    if (!arquivoB) {
        perror("Erro ao abrir floats_256_5.0f.dat");
        fclose(arquivoA);
        return 1;
    }

    total = (long long)DIMENSION * DIMENSION;  

    for (i = 0; i < total; i++) {
        value = (float)i;
        fwrite(&value, sizeof(float), 1, arquivoA);
    }

    for (i = 0; i < total; i++) {
        value = (float)(i + 1);
        fwrite(&value, sizeof(float), 1, arquivoB);
    }

    fclose(arquivoA);
    fclose(arquivoB);

    printf("Arquivos gerados com sucesso! (%lld elementos em cada matriz %d X %d)\n", 
           total, DIMENSION, DIMENSION);
    return 0;
}
