#include <stdio.h>
#include <stdlib.h>

int main(void) {
    FILE *fa, *fb;
    float value;
    int i;

    fa = fopen("BinFiles/matrizA.dat", "wb");
    if (!fa) {
        perror("Erro ao abrir matrizA.dat");
        return 1;
    }

    fb = fopen("BinFiles/matrizB.dat", "wb");
    if (!fb) {
        perror("Erro ao abrir matrizB.dat");
        fclose(fa);
        return 1;
    }

    value = 2.0f;
    for (i = 0; i <= 2048; i++) {
        fwrite(&value, sizeof(float), 1, fa);
    }

    // Escreve 128 floats = 5.0 em matrizB.dat
    value = 5.0f;
    for (i = 0; i <= 2048; i++) {
        fwrite(&value, sizeof(float), 1, fb);
    }

    fclose(fa);
    fclose(fb);

    printf("Arquivos gerados com sucesso!\n");
    return 0;
}
