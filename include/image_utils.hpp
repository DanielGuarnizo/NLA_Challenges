#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <string>
#include <vector>
#include "stb_image_write.h"  // STB library for writing images
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <fstream>


using namespace std; 
using namespace Eigen;

// Declaration of kernel matrices to apply convolutions 

extern MatrixXd H_av2;
extern MatrixXd H_sh2;
extern MatrixXd H_lap;

// Function to load an image from a file using stb_image
unsigned char* load_image(const char* input_image_path, int& width, int& height, int& channels);
    
void saveImage(const string& path, int width, int height, int channels, const vector<unsigned char>& image_data);

// Function to add noise to the image
Matrix<unsigned char, Dynamic, Dynamic, RowMajor> addNoiseToImage(const MatrixXd &original_image);

SparseMatrix<double> createConvolutionalMatrix(const int m, const int n, const MatrixXd& kernel);

void loadSolutionFromFile(const string& path_filename, const int n, const int m, const int channels, const string& result_image_path);

void exportVectorToMTX(const VectorXd& w, const std::string& filename);

void exportSparseMatrixToMTX(const SparseMatrix<double>& A, const std::string& filename);

void initializeKernels();

#endif // IMAGE_UTILS_H