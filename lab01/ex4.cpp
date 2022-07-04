#include "ex.h"
#include "util.h"
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

std::unordered_set<float> iteratef(float x0,float r,int initialIterations,int recordedIterations)
{
    float x=x0;
    while(initialIterations-- >0)
    {
        x=r*x*(1.0f-x);
    }
    std::unordered_set<float> set;
    while (recordedIterations-- >0)
    {
        set.insert(x);
        x=r*x*(1.0f-x);
    }
    return set;
}

std::unordered_set<double> iterated(double x0,double r,int initialIterations,int recordedIterations)
{
    float x=x0;
    while(initialIterations-- >0)
    {
        x=r*x*(1.0-x);
    }
    std::unordered_set<double> set;
    while (recordedIterations-- >0)
    {
        set.insert(x);
        x=r*x*(1.0-x);
    }
    return set;
}
void plotValuesd(double x0,double rStart,double rEnd,int n)
{
    int initial=1000;
    int record=1000;
    std::vector<double> x;
    std::vector<double> y;
    double dr=(rEnd-rStart)/n;
    for(double r=rStart;r<rEnd;r+=dr)
    {
        std::unordered_set<double> dValues = iterated(x0,r,initial,record);
        for(double val : dValues)
        {
            x.push_back(r);
            y.push_back(val);
        }
    }
    std::vector<float> fx(x.begin(),x.end());
    std::vector<float> fy(y.begin(),y.end());
    std::string title;
    title = "x0 = " + std::to_string(x0) + " double";

    util::plotPoints(fx,fy,title);
}
void plotValuesf(float x0,float rStart,float rEnd,int n)
{
    int initial=1000;
    int record=1000;
    std::vector<float> x;
    std::vector<float> y;
    float dr=(rEnd-rStart)/n;
    for(double r=rStart;r<rEnd;r+=dr)
    {
        std::unordered_set<float> dValues = iteratef(x0,r,initial,record);
        for(float val : dValues)
        {
            x.push_back(r);
            y.push_back(val);
        }
    }
    std::string title;
    title = "x0 = " + std::to_string(x0) + " float";


    util::plotPoints(x,y,title);
}
std::pair<int,float> ex4p3nofIterations(float x0)
{
    int maxI = 1000000;
    float minx = fabs(x0);
    float x = x0;
    int mini = 0;
    for(int i = 0;i<maxI;i++)
    {
        x=4*x*(1.0f-x);
        if(fabs(x)<minx)
        {
            minx=fabs(x);
            mini=i;
        }
    }
    return std::make_pair(mini, minx);
}
void ex4p1()
{
    plotValuesd(0.5,1,4,100);
    plotValuesd(0.242521,1,4,100);
    plotValuesd(0.77638,1,4,100);
    plotValuesd(0.1368936,1,4,100);
}

void ex4p2()
{
    plotValuesd(0.5,3.75,3.8,100);
    plotValuesf(0.5,3.75,3.8,100);
}
void ex4p3()
{

    std::pair<int,float> val1 = ex4p3nofIterations(0.5);
    std::pair<int,float> val2 = ex4p3nofIterations(0.242521);
    std::pair<int,float> val3 = ex4p3nofIterations(0.77638);
    std::pair<int,float> val4 = ex4p3nofIterations(0.1368936);
    std::pair<int,float> val5 = ex4p3nofIterations(0.501);

    std::cout << "osiągnięto najmniejszą wartość fabs(x) = " << val1.second << " po " << val1.first + 1 << " iteracjach dla  x0 = 0.5\n";
    std::cout << "osiągnięto najmniejszą wartość fabs(x) = " << val2.second << " po " << val2.first + 1 << " iteracjach dla  x0 = 0.242521\n";
    std::cout << "osiągnięto najmniejszą wartość fabs(x) = " << val3.second << " po " << val3.first + 1 << " iteracjach dla  x0 = 0.77638\n";
    std::cout << "osiągnięto najmniejszą wartość fabs(x) = " << val4.second << " po " << val4.first + 1 << " iteracjach dla  x0 = 0.1368936\n";
    std::cout << "osiągnięto najmniejszą wartość fabs(x) = " << val5.second << " po " << val5.first + 1 << " iteracjach dla  x0 = 0.501\n";

    std::vector<float> x;
    std::vector<float> y;
    for(float x0 = 0;x0<1;x0+=0.0001)
    {
        x.push_back(x0);
        y.push_back(ex4p3nofIterations(x0).first);
    }
    util::plotPoints(x,y,"");
}