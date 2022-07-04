#include <iostream>
#include <vector>
#include "util.h"
#include <iomanip>
#include "gnuplot_i.hpp"
#include "common.h"
#include <chrono>
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif


//gnuplot_i from https://code.google.com/archive/p/gnuplot-cpp/

std::ostream &operator<<(std::ostream &os, util::ffloat val) {
    os << std::fixed << std::showpoint << std::setprecision(val.precision) << val.value;
    return os;
}
std::ostream &operator<<(std::ostream &os, util::fdouble val) {
    os << std::fixed << std::showpoint << std::setprecision(val.precision) << val.value;
    return os;
}

namespace util {
    void plotLines(std::vector<float> x, std::vector<float> y, std::string title) {
        Gnuplot gp("lines");
        //gp.savetops("graph");
        gp.plot_xy(x, y, title);
        sleep(10);
    }
    void plotPoints(std::vector<float> x, std::vector<float> y, std::string title)
    {
        Gnuplot gp("points");
        //gp.savetops("graph");
        gp.plot_xy(x, y, title);
        sleep(10);
    }
    int64_t timeIt(std::function<void()> function, int times) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < times; i++)
            function();
        auto stop = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                stop - start).count();
    }

}

