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

// Function to load an image from a file using stb_image
unsigned char* load_image(const char* input_image_path, int& width, int& height, int& channels);
    
void saveImage(const string& path, int width, int height, int channels, const vector<unsigned char>& image_data);

void exportVectorToMTX(const VectorXd& w, const std::string& filename);

void exportSparseMatrixToMTX(const SparseMatrix<double>& A, const std::string& filename);


#endif // IMAGE_UTILS_H