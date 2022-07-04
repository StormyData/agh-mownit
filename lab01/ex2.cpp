#include "ex.h"
#include "util.h"
#include "common.h"
#include <vector>
#include <iostream>

void ex2p1() {
    const int N = 10000000;
    const float v = 0.53125f;
    const float vN = 5312500;
    std::vector tab(N,v);
    float sum = kahanSum(tab.data(), N);
    std::cout << "1)\n";
    std::cout << "\tsum = " << util::ffloat(sum, 1) << '\n';
    std::cout << "\tv*N = " << util::ffloat(vN, 1) << "\n";
    std::cout << "\tbłąd bezwzględny sum - v * N = " << util::ffloat(sum - vN, 1) << "\n";
    std::cout << "\tbłąd względny (sum - v*N)/(v*N) = " << util::ffloat((sum - vN) / vN, 4) << "\n";
}
void ex2p3() {
    const int chrono_N = 100;
    const int N = 10000000;
    const float v = 0.53125f;
    std::vector tab(N,v);
    std::cout << "3)\n";
    std::cout << "mierzenie czasu\n";

    int64_t timeRecursive = util::timeIt(std::bind(recursiveSum,tab.data(),0,N),chrono_N);
    int64_t timeKahan = util::timeIt(std::bind(kahanSum,tab.data(),N),chrono_N);

    std::cout << "\twykonano " << chrono_N << " powtórzeń\n";
    std::cout << "\tdodawanie kahana zajeło " << timeKahan << " ns\n";
    std::cout << "\tdodawanie rekurencyjne zajeło " << timeRecursive << " ns\n";
}
