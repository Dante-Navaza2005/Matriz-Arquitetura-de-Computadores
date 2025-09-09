#include <stdio.h>
#include <stdlib.h>

#define N 2048  // dimensão da matriz (NxN)

int main(void) {
    FILE *fa, *fb;
    float value;
    long total = (long)N * (long)N;  // número total de floats
    long i;

    fa = fopen("matrizA.dat", "wb");
    if (!fa) {
        perror("Erro ao abrir matrizA.dat");
        return 1;
    }

    fb = fopen("matrizB.dat", "wb");
    if (!fb) {
        perror("Erro ao abrir matrizB.dat");
        fclose(fa);
        return 1;
    }

    // Escreve matrizA: todos os elementos = 2.0
    value = 2.0f;
    for (i = 0; i < total; i++) {
        fwrite(&value, sizeof(float), 1, fa);
    }

    // Escreve matrizB: todos os elementos = 5.0
    value = 5.0f;
    for (i = 0; i < total; i++) {
        fwrite(&value, sizeof(float), 1, fb);
    }

    fclose(fa);
    fclose(fb);

    printf("Arquivos gerados com sucesso! (%dx%d floats)\n", N, N);
    return 0;
}
