#pragma once

#include <ios>
#include <vector>
#include <functional>

namespace util {
    struct ffloat {
        float value;
        int precision;

        ffloat(float val, int prec) : value(val), precision(prec) {}
    };
    struct fdouble {
        double value;
        int precision;

        fdouble(double val, int prec) : value(val), precision(prec) {}
    };

    void plotLines(std::vector<float> x, std::vector<float> y, std::string title);
    void plotPoints(std::vector<float> x, std::vector<float> y, std::string title);
    int64_t timeIt(std::function<void()>,int);
}

std::ostream &operator<<(std::ostream &, util::ffloat);
std::ostream &operator<<(std::ostream &, util::fdouble);