#include <stdio.h>
#include <stdlib.h>

#define N 2048  

int main(void) {
    FILE *arquivoA, *arquivoB;
    float value;
    long long total;
    long long i;

    arquivoA = fopen("matrizA.dat", "wb");
    if (!arquivoA) {
        perror("Erro ao abrir matrizA.dat");
        return 1;
    }

    arquivoB = fopen("matrizB.dat", "wb");
    if (!arquivoB) {
        perror("Erro ao abrir matrizB.dat");
        fclose(arquivoA);
        return 1;
    }

    total = (long long)N * N;  // número total de elementos (2048*2048)

    // Escreve matriz A (todos 2.0f)
    value = 2.0f;
    for (i = 0; i < total; i++) {
        fwrite(&value, sizeof(float), 1, arquivoA);
    }

    // Escreve matriz B (todos 5.0f)
    value = 5.0f;
    for (i = 0; i < total; i++) {
        fwrite(&value, sizeof(float), 1, arquivoB);
    }

    fclose(arquivoA);
    fclose(arquivoB);

    printf("Arquivos gerados com sucesso! (%lld elementos em cada matriz)\n", total);
    return 0;
}
