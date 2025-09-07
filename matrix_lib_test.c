#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include "matrix_lib.h" 

// Escreve simultaneamente na tela e no arquivo de relatório
static void dual_printf(FILE *report, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    if (report) {
        va_start(args, fmt);
        vfprintf(report, fmt, args);
        va_end(args);
        fflush(report);
    }
}

// Imprime até 256 elementos de uma matriz varrendo por colunas
static void print_matrix_limited(const char *label, const Matrix *m, FILE *report) {
    unsigned long int total = m->height * m->width;
    unsigned long int limit = total < 256UL ? total : 256UL;
    dual_printf(report, "\n%s (h=%lu, w=%lu) - até %lu elementos (ordem coluna):\n",
                label, m->height, m->width, limit);

    unsigned long int printed = 0;
    for (unsigned long int col = 0; col < m->width && printed < limit; col++) {
        for (unsigned long int row = 0; row < m->height && printed < limit; row++) {
            float v = m->rows[row * m->width + col];
            dual_printf(report, "%.2f ", v);
            printed++;
        }
    }
    dual_printf(report, "\n");
}

// Captura o modelo do processador via lscpu; fallback no macOS
static void print_cpu_model(FILE *report) {
    char buffer[4096];
    FILE *pipe = popen("lscpu 2>/dev/null | grep -i 'model name' | sed 's/.*: *//'", "r");
    if (pipe) {
        if (fgets(buffer, sizeof(buffer), pipe)) {
            // Remove newline
            buffer[strcspn(buffer, "\r\n")] = 0;
            dual_printf(report, "CPU Model: %s\n", buffer);
            pclose(pipe);
            return;
        }
        pclose(pipe);
    }

    // Fallback para macOS
    pipe = popen("sysctl -n machdep.cpu.brand_string 2>/dev/null", "r");
    if (pipe) {
        if (fgets(buffer, sizeof(buffer), pipe)) {
            buffer[strcspn(buffer, "\r\n")] = 0;
            dual_printf(report, "CPU Model: %s\n", buffer);
            pclose(pipe);
            return;
        }
        pclose(pipe);
    }
    dual_printf(report, "CPU Model: (não disponível)\n");
}

int getMatrixFromFile(char* path, Matrix* m){
    FILE* f = fopen(path, "rb");

    if(!f){
        printf("Erro de leitura");
        exit(1);
    }

     // Lê todos os elementos de uma vez
    size_t total = m->height * m->width;
    size_t read = fread(m->rows, sizeof(float), total, f);
    fclose(f);

    if (read != m->height * m->width) {
        fprintf(stderr, "Erro: número incorreto de elementos lidos de %s\n", path);
        return 0;
    }


    return 1;
}

int saveMatrix(char* path, Matrix *m){
    FILE* f = fopen(path, "wb");

    if(!f){
        printf("Erro de leitura");
        exit(1);
    }

    size_t written = fwrite(m->rows, sizeof(float), m->height * m->width, f);
    fclose(f);

    if (written != m->height * m->width) {
        fprintf(stderr, "Erro: número incorreto de elementos escritos em %s\n", path);
        return 0;
    }
    return 1;
}


int main(int argc, char *argv[]) {
    if (argc != 10) {
        printf("Uso: ./matrix_lib_test <escalar> <hA> <wA> <hB> <wB> <arquivoA> <arquivoB> <arquivoA_result> <arquivoC>\n");
        return 1;
    }

    struct timeval overall_t1, overall_t2, t_start, t_stop;
    gettimeofday(&overall_t1, NULL);

    FILE *report = fopen("relatorio.txt", "w");
    if (!report) {
        perror("relatorio.txt");
        return 1;
    }

    // Lendo os argumentos da linha de comando
    float num_esc = atof(argv[1]);
    unsigned long int heightA = atol(argv[2]);
    unsigned long int widthA = atol(argv[3]);
    unsigned long int heightB = atol(argv[4]);
    unsigned long int widthB = atol(argv[5]);

    char* fileA = argv[6];
    char* fileB = argv[7];
    char* fileA_r = argv[8];
    char* fileC = argv[9];

    if (widthA != heightB) { 
        dual_printf(report, "O num de colunas de A (%lu) é diferente do num de linhas de B (%lu).\n", widthA, heightB);
        fclose(report);
        return 1;
    }

    Matrix mA = {heightA, widthA, malloc(sizeof(float) * heightA * widthA)};
    Matrix mB = {heightB, widthB, malloc(sizeof(float) * heightB * widthB)};
    Matrix mC = {heightA, widthB, malloc(sizeof(float) * heightA * widthB)};

    // Inicializar a matriz C com zeros apenas
    for (int i = 0; i < heightA * widthB; i++) {
        mC.rows[i] = 0.0f;
    }

    // Populando matriz A e B dos arquivos binários
    if (!getMatrixFromFile(fileA, &mA)){
        return 1;
    } 
    print_matrix_limited("Matriz A", &mA, report);

    if (!getMatrixFromFile(fileB, &mB)){
        return 1;
    } 
    print_matrix_limited("Matriz B", &mB, report);

    // Medindo scalarMatrixMult
    gettimeofday(&t_start, NULL);
    if (!scalarMatrixMult(num_esc, &mA)) {
        dual_printf(report, "Erro ao calcular a multiplicação escalar de A\n");
        fclose(report);
        return 1;
    }
    gettimeofday(&t_stop, NULL);
    dual_printf(report, "\nTempo da scalarMatrixMult: %.3f ms\n", timedifference_msec(t_start, t_stop));

    if (!saveMatrix(fileA_r, &mA)) return 1;
    dual_printf(report, "\nMatriz A dps da multiplicacao escalar por %.2f:", num_esc);
    print_matrix_limited("", &mA, report);

    // Medindo matrixMatrixMult
    gettimeofday(&t_start, NULL);
    if (!matrixMatrixMult(&mA, &mB, &mC)) {
        dual_printf(report, "Erro ao multiplicar as matrizes A e B\n");
        fclose(report);
        return 1;
    }
    gettimeofday(&t_stop, NULL);
    dual_printf(report, "\nTempo da matrixMatrixMult: %.3f ms\n", timedifference_msec(t_start, t_stop));

    // Salvando a matriz C
    if (!saveMatrix(fileC, &mC)) return 1;
    print_matrix_limited("Matriz C dps de multiplicar A e B", &mC, report);

    // CPU model
    dual_printf(report, "\n");
    print_cpu_model(report);

    free(mA.rows);
    free(mB.rows);
    free(mC.rows);

    gettimeofday(&overall_t2, NULL);  // fim do programa
    dual_printf(report, "\nTempo total do programa: %.3f ms\n", timedifference_msec(overall_t1, overall_t2));

    fclose(report);


    return 0;
}
