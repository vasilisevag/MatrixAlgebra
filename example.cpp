#include "matrix.h"

int main(){
    mat::DenseMatrix<double, 3, 3> rgb2yuv {{0.299, 0.587, 0.114},
                                            {-0.147, -0.288, 0.436},
                                            {0.615, -0.514, -0.10}};

    mat::Vector<int, 3> rgbPixel = {255, 0, 0}; // color = red

    auto yuvPixel = rgb2yuv * rgbPixel;
    std::cout << T(yuvPixel);
}