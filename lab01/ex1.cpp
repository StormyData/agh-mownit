#include "ex.h"
#include "util.h"
#include "common.h"
#include <chrono>
#include <iostream>
#include <vector>


const int chrono_N = 100;
const int N = 10000000;
const int n = 25000;
const float v = 0.53125f;
const float vN = 5312500;


void ex1p1and2() {
    std::vector<float> tab(N, v);
    float sum = iterativeSum(tab.data(), N);
    std::cout << "1,2)\n";
    std::cout << "\tsum = " << util::ffloat(sum, 1) << '\n';
    std::cout << "\tv*N = " << util::ffloat(vN, 1) << "\n";
    std::cout << "\tbłąd bezwzględny sum - v * N = " << util::ffloat(sum - vN, 1) << "\n";
    std::cout << "\tbłąd względny (sum - v*N)/(v*N) = " << util::ffloat((sum - vN) / vN, 4) << "\n";
}


void ex1p3() {
    float *tab = new float[N];
    for (int i = 0; i < N; i++)
        tab[i] = v;
    float sum = 0;
    std::cout << "3)\n";
    std::cout << "\traport wzrostu błędu\n";
    std::cout << "\titeracja / wartość obliczona - wartość oczekiwana\n";
    std::vector<float> x;
    std::vector<float> y;
    for (int i = 0; i < N; i++) {
        sum += tab[i];
        if ((i + 1) % n == 0) {
            float err = sum - v * (i + 1);
            std::cout << "\t" << i + 1 << "\t\t" << util::ffloat(err, 7) << "\n";
            x.push_back(i + 1);
            y.push_back(err);
        }

    }
    delete[] tab;
    util::plotLines(x, y, "error");
}

void ex1p4and5() {
    std::vector<float> tab(N, v);
    float sum = recursiveSum(tab.data(), 0, N);
    std::cout << "4,5)\n";
    std::cout << "\tliczenie rekurencyjnie\n";
    std::cout << "\tsum = " << util::ffloat(sum, 1) << "\n";
    std::cout << "\tv*N = " << util::ffloat(vN, 1) << "\n";
    std::cout << "\tbłąd bezwzględny sum - v*N = " << util::ffloat(sum - vN, 1) << "\n";
    std::cout << "\tbład względny (sum - v*N)/(v*N) = " << util::ffloat((sum - vN) / vN, 7) << "\n";
}

void ex1p6() {
    std::vector<float> tab(N, v);
    std::cout << "6)\n";
    std::cout << "mierzenie czasu\n";

    int64_t timeIterative = util::timeIt(std::bind(iterativeSum,tab.data(),N),chrono_N);
    int64_t timeRecursive = util::timeIt(std::bind(recursiveSum,tab.data(),0,N),chrono_N);

    std::cout << "\twykonano " << chrono_N << " powtórzeń\n";
    std::cout << "\tdodawanie iterowane zajeło " << timeIterative << " ns\n";
    std::cout << "\tdodawanie rekurencyjne zajeło " << timeRecursive << " ns\n";
}
void ex1p7() {
    float tab[] = {1000000000.0f,  1000000000.0f,0.5f, 0.5f,100000000.0f,  -100000001.0f,
                   -0.5f, 0.5f,-10000000.0f,  10000001.0f,-0.5f, 0.5f,-10000000.0f,
                   1000001.0f,0.05f, -0.5f};
    int len = sizeof(tab)/sizeof(tab[0]);
    float exactSum = 1991000001.55f;
    float sum = recursiveSum(tab, 0, len);
    std::cout << "7)\n";
    std::cout << "\tprzykład danych, dla których rekurencja ma niezerowy błąd\n";
    std::cout << "\twartość tablicy\n";
    for (int i = 0; i < len; i++)
        std::cout << "\ttab[" << i << "] = " << util::ffloat(tab[i],7) << "\n";
    std::cout << "\twartość wyliczona = " << util::ffloat(sum, 7) << "\n";
    std::cout << "\twartość dokładna = " << util::ffloat(exactSum, 7) << "\n";
    std::cout << "\twartość błędu (wyliczone - dokładne) = " << util::ffloat(sum - exactSum, 7) << "\n";
}

