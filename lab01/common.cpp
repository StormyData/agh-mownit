#include "common.h"

float kahanSum(const float *tab, const int n) {
    float sum = 0.0f;
    float err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float y = tab[i] - err;
        float temp = sum + y;
        err = (temp - sum) - y;
        sum = temp;
    }
    return sum;
}

float recursiveSum(const float *tab, const int from, const int to) {
    if (to <= from)
        return 0;
    if (to == from + 1)
        return tab[from];
    int mid = from + (to - from) / 2;
    return recursiveSum(tab, from, mid) + recursiveSum(tab, mid, to);
}

float iterativeSum(const float *tab, const int N) {
    float sum = 0;
    for (int i = 0; i < N; i++)
        sum += tab[i];
    return sum;
}