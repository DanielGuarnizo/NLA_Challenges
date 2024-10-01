#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <string>
#include <vector>
#include "stb_image_write.h"  // STB library for writing images
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std; 
using namespace Eigen;

// Declaration of kernel matrices to apply convolutions 

extern MatrixXd H_av2;
extern MatrixXd H_sh2;
    
void saveImage(const string& path, int width, int height, int channels, const vector<unsigned char>& image_data);

SparseMatrix<double> createConvolutionalMatrix(const int m, const int n, const MatrixXd& kernel);

void initializeKernels();

#endif // IMAGE_UTILS_H