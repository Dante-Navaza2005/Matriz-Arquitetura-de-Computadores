#include <stdio.h>

int main() {
    int avx  = __builtin_cpu_supports("avx");
    int avx2 = __builtin_cpu_supports("avx2");
    int fma  = __builtin_cpu_supports("fma");

    printf("AVX:  %s\n", avx  ? "suportado" : "não suportado");
    printf("AVX2: %s\n", avx2 ? "suportado" : "não suportado");
    printf("FMA:  %s\n", fma  ? "suportado" : "não suportado");

    return 0;
}
