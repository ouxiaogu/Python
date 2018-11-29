#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <iterator>
#include <sstream>
#include <utility>
#include <libgen.h>
#include <cstdio>
#include <functional>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
using namespace std;
using namespace boost::accumulators;


std::vector <double> gradient = { 3.72381939e-02,   2.20510931e-02,   6.78594406e-02,
         3.72381939e-02, 2.20510931e-02, 6.78594406e-02, 7.36125512e-02, 4.67879807e-02, 8.60414202e-02,
    3.59425545e-02, 8.89476123e-03, 2.87074488e-04, 8.73628142e-04, 9.32779382e-04, 3.89541830e-03,
    2.18272907e-03, 1.23104585e-03, 5.69371361e-04, 1.99977285e-04, 1.70850525e-04, 8.61731086e-05,
    3.16111359e-05, 6.32469903e-04, 2.74752232e-06, 7.00564321e-04, 2.42549967e-04, 7.18512055e-05,
    4.17510011e-04, 4.38682973e-05, 4.04680943e-05, 3.34822493e-05, 3.92209594e-06, 3.33340424e-06,
    1.35656520e-05, 4.99800945e-06, 1.75421376e-05, 3.04015224e-06, 3.01437605e-06, 2.59021351e-05,
    1.03564354e-05, 1.56100393e-05, 6.31614335e-06, 1.70179171e-07, 1.44091441e-06, 4.66400672e-06,
    4.81974159e-06, 1.74554600e-05, 2.54090659e-06, 1.20191787e-05, 1.91222383e-05, 1.46297970e-05,
    7.04003487e-07, 7.20127430e-06, 1.22969647e-05, 9.30846214e-06, 2.63896602e-05, 1.98528449e-05,
    3.04527692e-05, 4.17143161e-05, 1.98888595e-05, 3.44119280e-07, 9.53279592e-06, 5.59613613e-06,
    4.37915823e-06, 7.28224057e-06, 1.28666125e-06, 1.27327688e-05, 1.94775852e-05, 1.73946296e-06,
    1.20575077e-05, 4.46496243e-06, 2.09017441e-05, 1.71017062e-05, 1.97038347e-05, 2.39629581e-05,
    6.94169778e-06, 1.62270065e-05, 6.10197505e-05, 7.66869708e-06, 6.28422238e-05, 1.33742681e-05,
    3.81911289e-06, 9.60250672e-07, 1.05918344e-07, 5.01230854e-06, 1.25014761e-05, 1.12932910e-05,
    2.04433419e-06, 8.40386009e-06, 8.07688063e-06, 3.60238816e-06, 1.03056802e-05, 2.05837957e-05,
    1.09141764e-05, 2.02428888e-05, 2.22164782e-05, 1.21485892e-07, 8.68103286e-06, 1.26108934e-05,
    1.38385735e-05, 1.23346634e-05, 4.56318564e-06, 1.51187078e-05, 2.38895265e-05, 7.59926006e-06,
    3.75926022e-06, 6.83239461e-07, 7.06169570e-06, 6.00681478e-06, 6.72487245e-07, 1.92173402e-06,
    1.34355042e-06, 1.26587965e-06, 2.59111227e-06, 7.91838686e-07, 5.89250712e-06, 4.42679198e-06,
    2.55868833e-06, 5.08900794e-06, 1.60863001e-06, 8.14277111e-07, 1.69879512e-06, 1.05628002e-06,
    4.81360130e-06, 6.49971407e-06, 6.96404235e-06, 1.07888898e-05, 9.62865776e-06, 1.87153142e-05,
    7.53263033e-07, 6.19622299e-06, 2.15935741e-05, 2.37675204e-05, 3.42946591e-06, 1.59841998e-05,
    7.21086047e-06, 1.52837010e-05, 3.31519020e-06, 1.76668102e-05, 1.49424868e-05, 5.02553099e-06,
    1.74625942e-06, 4.04326895e-06, 1.87153208e-06, 2.58438446e-06, 7.74520726e-08, 3.02825956e-06,
    2.18032598e-06, 8.48295371e-06, 5.31799671e-06, 4.34556578e-06, 8.18864597e-06, 2.24444007e-06,
    2.48692413e-06, 3.89600405e-06, 1.17926339e-06, 1.39351172e-06, 5.41469875e-07, 9.53073177e-07,
    6.65203932e-07, 3.10473334e-06, 2.48504154e-06, 5.51722865e-07, 3.59033331e-06, 2.54491114e-06,
    2.88849027e-06, 1.05095148e-05, 1.79325141e-05, 1.65017562e-05, 1.84237079e-06, 1.63607544e-05,
    1.58578707e-05, 1.09784771e-05, 3.55858665e-06
};

std::vector <double> gradient2 = {
    0.0372382, 0.0220511, 0.0678594, 0.0736126, 0.046788, 0.0860414, 0.0359426, 0.00889476, 0.000287074,
    0.000873628, 0.000932779, 0.00389542, 0.00218273, 0.00123105, 0.000569371, 0.000199977,
    0.000170851, 8.61731e-05, 3.16111e-05, 0.00063247, 2.74752e-06, 0.000700564, 0.00024255,
    7.18512e-05, 0.00041751, 4.38683e-05, 4.04681e-05, 3.34822e-05, 3.9221e-06, 3.3334e-06,
    1.35657e-05, 4.99801e-06, 1.75421e-05, 3.04015e-06, 3.01438e-06, 2.59021e-05, 1.03564e-05,
    1.561e-05, 6.31614e-06, 1.70179e-07, 1.44091e-06, 4.66401e-06, 4.81974e-06, 1.74555e-05,
    2.54091e-06, 1.20192e-05, 1.91222e-05, 1.46298e-05, 7.04004e-07, 7.20127e-06, 1.2297e-05,
    9.30846e-06, 2.63897e-05, 1.98528e-05, 3.04528e-05, 4.17143e-05, 1.98889e-05, 3.44119e-07,
    9.5328e-06, 5.59614e-06, 4.37916e-06, 7.28224e-06, 1.28666e-06, 1.27328e-05, 1.94776e-05,
    1.73946e-06, 1.20575e-05, 4.46496e-06, 2.09017e-05, 1.71017e-05, 1.97038e-05, 2.3963e-05,
    6.9417e-06, 1.6227e-05, 6.10198e-05, 7.6687e-06, 6.28422e-05, 1.33743e-05, 3.81911e-06,
    9.60251e-07, 1.05918e-07, 5.01231e-06, 1.25015e-05, 1.12933e-05, 2.04433e-06, 8.40386e-06,
    8.07688e-06, 3.60239e-06, 1.03057e-05, 2.05838e-05, 1.09142e-05, 2.02429e-05, 2.22165e-05,
    1.21486e-07, 8.68103e-06, 1.26109e-05, 1.38386e-05, 1.23347e-05, 4.56319e-06, 1.51187e-05,
    2.38895e-05, 7.59926e-06, 3.75926e-06, 6.83239e-07, 7.0617e-06, 6.00681e-06, 6.72487e-07,
    1.92173e-06, 1.34355e-06, 1.26588e-06, 2.59111e-06, 7.91839e-07, 5.89251e-06, 4.42679e-06,
    2.55869e-06, 5.08901e-06, 1.60863e-06, 8.14277e-07, 1.6988e-06, 1.05628e-06, 4.8136e-06,
    6.49971e-06, 6.96404e-06, 1.07889e-05, 9.62866e-06, 1.87153e-05, 7.53263e-07, 6.19622e-06,
    2.15936e-05, 2.37675e-05, 3.42947e-06, 1.59842e-05, 7.21086e-06, 1.52837e-05, 3.31519e-06,
    1.76668e-05, 1.49425e-05, 5.02553e-06, 1.74626e-06, 4.04327e-06, 1.87153e-06, 2.58438e-06,
    7.74521e-08, 3.02826e-06, 2.18033e-06, 8.48295e-06, 5.318e-06, 4.34557e-06, 8.18865e-06,
    2.24444e-06, 2.48692e-06, 3.896e-06, 1.17926e-06, 1.39351e-06, 5.4147e-07, 9.53073e-07,
    6.65204e-07, 3.10473e-06, 2.48504e-06, 5.51723e-07, 3.59033e-06, 2.54491e-06, 2.88849e-06,
    1.05095e-05, 1.79325e-05, 1.65018e-05, 1.84237e-06, 1.63608e-05, 1.58579e-05, 1.09785e-05,
    3.55859e-06
};

template<typename T>
std::string vec2str(const std::vector<T>& src)
{
    if(src.empty()){
        cerr << "vec2str, input map can not be empty!\n";
        return string("");
    }

    stringstream ss;
    std::copy(src.begin(), src.end(), std::ostream_iterator<T>(ss, ", "));
    string dst = ss.str();
    dst = dst.substr(0, dst.length()-2); // get rid of the trailing ", "
    return dst;
}
// instantiate
template std::string vec2str(const std::vector<double>&);
template std::string vec2str(const std::vector<float>&);
template std::string vec2str(const std::vector<string>&);
template std::string vec2str(const std::vector<int>&);

std::vector<double>
pad(const std::vector<double> &src, const std::pair<size_t, size_t> &padWidth,
    const std::string mode, const std::map<std::string, double> &options)
{
    size_t preWidth = padWidth.first, postWidth = padWidth.second;
    std::vector<double> padArr(src.size()+preWidth+postWidth, 0);

    double preValue, postValue;
    if ( mode.compare("edge") == 0 ) {
        preValue = src[0];
        postValue = src[src.size() - 1];
    } else { // default as "constant" mode
        preValue = postValue = options.find("constant_value")->second;
    }

    for (size_t i = 0; i < padArr.size(); ++i)
    {
        if(i < preWidth)
            padArr[i] = preValue;
        else if(i >= padArr.size() - postWidth)
            padArr[i] = postValue;
        else
            padArr[i] = src[i - preWidth];
    }
    return padArr;
}

std::vector<double> apply1DFilter(const std::vector<double> &src, const std::vector<double> &flt)
{
    size_t halfFltSz = flt.size()/2;
    std::pair<size_t, size_t> padWidth(halfFltSz, halfFltSz);
    std::vector<double> padArr = pad(src, padWidth, "edge", std::map<std::string, double>());

    std::vector<double> dst;
    for (size_t i = 0; i < src.size(); ++i)
    {
        std::vector<double> temp(flt.size(), 0.);
        std::transform(flt.begin(), flt.end(), padArr.begin()+i, temp.begin(), std::multiplies<double>());
        double v = std::accumulate(temp.begin(), temp.end(), 0.0);
        dst.push_back(v);
    }
    return dst;
}

template <typename T>
std::vector<T> crossProductVec3(const std::vector<T> &a_, const std::vector<T> &b_)
{   
    assert(a_.size() == b_.size());
    assert(a_.size() == 2 || a_.size() == 3);
    std::vector<T> a;
    std::vector<T> b;
    if (a_.size() == 2) {
        a.resize(3);
        b.resize(3);
        std::copy(a_.begin(), a_.end(), a.begin());
        std::copy(b_.begin(), b_.end(), b.begin());
        a[2] = 0;
        b[2] = 0;
    } else {
        a = a_;
        b = b_;
    }
    cout << "a: " << vec2str(a) << "\n";
    cout << "b: " << vec2str(b) << "\n";
    std::vector<T> dst;
    dst.resize(3);
    
    dst[0] = a[1] * b[2] - a[2] * b[1];
    dst[1] = a[0] * b[2] - a[2] * b[0];
    dst[2] = a[0] * b[1] - a[1] * b[0];
    return dst;
}

std::vector<double> computeGradient(const std::vector<double> &src)
{
    size_t n = src.size();
    if (n < 3) {
        printf("Input vector size %zu < 3, return gradient as itself\n", n);
        return src;
    }

    std::vector<double> dst;
    for (size_t i = 0; i < n; ++i) {
        double temp;
        if (i == 0) {
            temp = -1.5 * src[0] + 2 * src[1] - 0.5 * src[2];
        } else if (i == n - 1) {
            temp = 1.5 * src[n - 1] - 2 * src[n - 2] + 0.5 * src[n - 3];
        } else {
            temp = (src[i + 1] - src[i - 1]) / 2.;
        }
        dst.push_back(temp);
    }
    return dst;
}

bool calcHistogram(const std::vector<double> &src, std::vector<int> &hist,
             std::vector<double> &bin_edges, int nbins_)
{
    size_t nbins = size_t(nbins_ > 0 ? nbins_ : 10);
    if (src.size() < nbins) {
        printf("Input vector size %zu < bin size %zu, can't calculate histogram!\n", src.size(), nbins);
        return false;
    }

    double vmin = *std::min_element(src.begin(), src.end());
    double vmax = *std::max_element(src.begin(), src.end());

    hist.clear();
    bin_edges.clear();
    hist = std::vector<int>(nbins, 0);
    bin_edges.resize(nbins + 1);
    for (size_t jj = 0; jj < nbins; ++jj) {
        bin_edges[jj] = vmin + (vmax - vmin) / nbins * jj;
    }
    bin_edges[nbins] = vmax;

    for (size_t ii = 0; ii < src.size(); ++ii) {
        if (src[ii] == bin_edges[nbins]) {
            hist[nbins - 1] += 1;
            continue;
        }
        for (size_t jj = 0; jj < nbins; ++jj) {
            if (bin_edges[jj] <= src[ii] && src[ii] < bin_edges[jj+1]) {
                hist[jj] += 1;
                continue;
            }
        }
    }
    return true;
}

double autoThres(const std::vector<double> &arr, int nbins=10, const std::string &mode="")
{
    std::vector<int> hist;
    std::vector<double> bin_edges;
    calcHistogram(arr, hist, bin_edges, nbins);
    vector<int>::iterator itMax = max_element(hist.begin(), hist.end());
    int idxMax = std::distance(hist.begin(), itMax);
    double lower_bound = bin_edges[idxMax], upper_bound = bin_edges[idxMax + 1];

    std::vector<double> largestBinValues;
    for (auto val: arr) {
        if(lower_bound <= val && val < upper_bound)
            largestBinValues.push_back(val); 
    }
    cout << "hist: " << vec2str(hist) << "\n";
    cout << "bins: " << vec2str(bin_edges) << "\n";

    accumulator_set<double, features<tag::mean, tag::median> > acc;
    std::for_each(std::begin(largestBinValues), std::end(largestBinValues), std::ref(acc));

    double thresh;
    if(mode.compare("median") == 0)
        thresh = median(acc);
    else
        thresh = mean(acc);
    return thresh;
}

int main()
{
    /* // concatenate string
    char buf[] = __FILE__;
    std::cout << "current file path:" << __FILE__ << ", "
              << basename(buf)  << "\n";
    string basedir("/gpfs/DEV/FEM/peyang/release/E8.0/MOD10660/MXP_job1/h/cache/dummydb/result/MXP/job1/");
    const char* relpath = "Average301result1/1_imageset.xml";
    const char* filepath =  (basedir + relpath).c_str();
    printf("path: %s\n", filepath);
    */
    
    // print vector

    vector<double> dVec= {0.1, 0.2, 0.9};
    cout << "dVec: " << vec2str(dVec) << "\n";
    
    int a, b, c;
    a = b = c = 5;
    cout << a << c << "\n";
    
    int halfFltSz = dVec.size()/2;
    std::pair<int, int> padWidth(halfFltSz, halfFltSz);
    cout << padWidth.first << endl;
    
    vector<double> flt= {0.25, 0.5, 0.25};
    vector<double> smoothed = apply1DFilter(dVec, flt);
    cout << "smoothed: " << vec2str(smoothed) << "\n";
    
    if ( 3 < a && a < 6)
        cout << a << "\n";
        
    dVec[1] = -0.2;
    auto op_fabs = [&](double val) {return fabs(val);};
    std::transform(dVec.begin(), dVec.end(), dVec.begin(), op_fabs);
    cout << "dVec: " << vec2str(dVec) << "\n";
    
    //std::reverse(dVec.begin(), dVec.end());
    //cout << "dVec: " << vec2str(dVec) << "\n";
    
    std::vector<double> crossvector = crossProductVec3(vector<double>({-0.93554688,  0.96905518}), vector<double>({0.89296571,  0.4501247}));
    double NeighborParalism = std::sqrt(std::inner_product(std::begin(crossvector), std::end(crossvector), std::begin(crossvector), 0.0)) / 1.34696543736;
    cout << "ret: " << vec2str(crossvector) << "\n" << NeighborParalism << "\n";
    
    printf("FN(missing) rate = %.3f%%\n", 34.);
    
    // vector<double> gradient = computeGradient(paralism);
    //cout << "gradient: " << vec2str(gradient) << "\n";
    double thres = autoThres(gradient);
    cout << "thres: " << thres << "\n";
    
    return 1;
}