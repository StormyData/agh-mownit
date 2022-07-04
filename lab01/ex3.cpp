#include "ex.h"
#include "util.h"
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <iostream>

float ex3dzetaForward(float s, int n) {
    float sum = 0;
    for(int k=1;k<=n;k++)
        sum += powf(k,-s);
    return sum;
}


float ex3dzetaReverse(float s, int n) {
    float sum = 0;
    for(int k=n;k>0;k--)
        sum += powf(k,-s);
    return sum;
}

float ex3etaForward(float s, int n) {
    float sum = 0;
    for(int k=1;k<=n;k++)
        sum += ((k&1) * 2 - 1) * powf(k,-s);
    return sum;
}

float ex3etaReverse(float s, int n) {
    float sum = 0;
    for(int k=n;k>0;k--)
        sum += ((k&1) * 2 - 1) * powf(k,-s);
    return sum;
}

double ex3dzetaForwardd(double s, int n)
{
    double sum = 0;
    for(int k=1;k<=n;k++)
        sum += pow(k,-s);
    return sum;
}

double ex3dzetaReversed(double s, int n)
{
    double sum = 0;
    for(int k=n;k>0;k--)
        sum += pow(k,-s);
    return sum;
}

double ex3etaForwardd(double s, int n)
{
    double sum = 0;
    for(int k=1;k<=n;k++)
        sum += ((k&1) * 2 - 1) * pow(k,-s);
    return sum;
}

double ex3etaReversed(double s, int n)
{
    double sum = 0;
    for(int k=n;k>0;k--)
        sum += ((k&1) * 2 - 1) * pow(k,-s);
    return sum;
}

void ex3()
{
    std::vector<double> exactDzetaValues = {1.6449340668482264364724151666460251892189499012067984377355582293,
                                            1.1094105145864533574507599836227330090757847120663655821473,
                                            1.0369277551433699263313654864570341680570809195019128119741926779,
                                            1.0072276664807171147390649245483405824525529306712675407825,
                                            1.0009945751278180853371459589003190170060195315644775172577889946};
    std::vector<double> exactEtaValues = {0.8224670334241132182362075833230125946094749506033992188677791146,
                                          0.9346933439191250729260744938015563076817822223823495454962,
                                          0.9721197704469093059356551435534695325535133620330432612258056355,
                                          0.9935270006616197875745232836798844665089208968326915333495,
                                          0.9990395075982715656392218456993418314259296496668906471068};
    std::vector<double> sValues = {2, 3.6667, 5, 7.2, 10};
    std::vector<int> nValues = {50, 100, 200, 500, 1000};

    std::cout << "funkcja dzeta do przodu float s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::ffloat(ex3dzetaForward((float) s, n), 14) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "funkcja dzeta do tyłu float s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::ffloat(ex3dzetaReverse((float)s,n),14) << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "funkcja dzeta do przodu double s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3dzetaForwardd(s, n), 14) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "funkcja dzeta do tyłu double s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3dzetaReversed(s, n), 14) << "\t";
        }
        std::cout << "\n";
    }


    std::cout << "funkcja eta do przodu float s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::ffloat(ex3etaForward((float)s,n),14) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "funkcja eta do tyłu float s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::ffloat(ex3etaReverse((float)s,n),14) << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "funkcja eta do przodu double s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3etaForwardd(s, n), 14) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "funkcja eta do tyłu double s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3etaReversed(s, n), 14) << "\t";
        }
        std::cout << "\n";
    }



    std::cout<<"\n\n\n";



    std::cout << "funkcja dzeta do przodu float - exact s\\n\n";
    for(int i=0;i<sValues.size();i++)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3dzetaForward((float) sValues[i], n) - exactDzetaValues[i], 14) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "funkcja dzeta do tyłu float - exact  s\\n\n";
    for(int i=0;i<sValues.size();i++)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3dzetaReverse((float)sValues[i],n) - exactDzetaValues[i],14) << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "funkcja dzeta do przodu double - exact  s\\n\n";
    for(int i=0;i<sValues.size();i++)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3dzetaForwardd(sValues[i], n) - exactDzetaValues[i], 14) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "funkcja dzeta do tyłu double - exact  s\\n\n";
    for(int i=0;i<sValues.size();i++)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3dzetaReversed(sValues[i], n) - exactDzetaValues[i], 14) << "\t";
        }
        std::cout << "\n";
    }


    std::cout << "funkcja eta do przodu float - exact  s\\n\n";
    for(int i=0;i<sValues.size();i++)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3etaForward((float)sValues[i],n) - exactEtaValues[i],14) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "funkcja eta do tyłu float - exact  s\\n\n";
    for(int i=0;i<sValues.size();i++)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3etaReverse((float)sValues[i],n) - exactEtaValues[i],14) << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "funkcja eta do przodu double - exact  s\\n\n";
    for(int i=0;i<sValues.size();i++)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3etaForwardd(sValues[i], n) - exactEtaValues[i], 14) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "funkcja eta do tyłu double - exact  s\\n\n";
    for(int i=0;i<sValues.size();i++)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3etaReversed(sValues[i], n) - exactEtaValues[i], 14) << "\t";
        }
        std::cout << "\n";
    }



    std::cout<<"\n\n\n";




    std::cout << "funkcja dzeta do przodu double - float s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3dzetaForwardd(s,n)-ex3dzetaForward((float) s, n), 14) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "funkcja dzeta do tyłu double - float s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3dzetaReversed(s,n)-ex3dzetaReverse((float)s,n),14) << "\t";
        }
        std::cout << "\n";
    }



    std::cout << "funkcja eta do przodu double - float s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3etaForwardd(s,n)-ex3etaForward((float)s,n),14) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "funkcja eta do tyłu double - float s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3etaReversed(s,n)-ex3etaReverse((float)s,n),14) << "\t";
        }
        std::cout << "\n";
    }



    std::cout <<"\n\n\n";



    std::cout << "funkcja dzeta (przód - tył) float s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::ffloat(ex3dzetaForward((float) s, n) - ex3dzetaReverse((float)s,n), 14) << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "funkcja dzeta (przód - tył) double s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3dzetaForwardd(s, n) - ex3dzetaReversed(s, n), 14) << "\t";
        }
        std::cout << "\n";
    }


    std::cout << "funkcja eta (przód - tył) float s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::ffloat(ex3etaForward((float)s,n) - ex3etaReverse((float)s,n),14) << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "funkcja eta (przód - tył) double s\\n\n";
    for(double s : sValues)
    {
        for(int n : nValues)
        {
            std::cout << util::fdouble(ex3etaForwardd(s, n) - ex3etaReversed(s, n), 14) << "\t";
        }
        std::cout << "\n";
    }
}