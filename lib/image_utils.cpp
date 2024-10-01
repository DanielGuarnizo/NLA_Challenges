#include "/home/jellyfish/shared-folder/Challenge_1_NLA/include/stb_image_write.h"
#include "../include/image_utils.hpp"
#include <iostream>
#include <vector>

// libraries to create Eigen matrices 
#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;
using namespace std;

// Initiliaze the Kernels matrix for convolutions
//& H_av2
double constantValue = 1.0 / 9.0;
MatrixXd H_av2 = MatrixXd::Constant(3, 3, constantValue);

//& H_sh2
MatrixXd H_sh2; // Define H_sh2 here

// initialize kernels 
void initializeKernels() {
    H_sh2 = MatrixXd::Zero(3, 3);  // Initialize to zero
    H_sh2 << 0.0, -3.0, 0.0,
             -1.0, 9.0, -3.0,
             0.0, -1.0, 0.0;
}


// Initialize functions
void saveImage(const string& path, int width, int height, int channels, const vector<unsigned char>& image_data) {
    // Save the image as PNG using stb_image_write
    if (stbi_write_png(path.c_str(), width, height, channels, image_data.data(), width * channels) == 0) {
        cerr << "Error: Could not save the image to " << path << endl;
    } else {
        cout << "Image saved successfully to " << path << endl;
    }
}

SparseMatrix<double> createConvolutionalMatrix(const int m, const int n, const MatrixXd& kernel){

     // Initialize the A_1 matrix
    SparseMatrix<double> A_1(m * n, m * n);
        //% Using Sparse matrix significally reduce the memory footprint, given that most of the values are zero
        //% Computation such as matrix multiplication are optimize for Sparse matrices and then more efficient to use
    vector<Triplet<double>> tripletList;
        //% we use this to store only the non-zero values of the sparse matrix in the following way (row, column, value)

    // variable to count the number of non zero entries of the sparse matrix
    int count = 0;

    // Fill sparse matrix A_1
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int index = (i * n) + j; // specify the row index in the sparse matrix
            for (int ki = 0; ki < 3; ki++) {
                for (int kj = 0; kj < 3; kj++) {
                    int imgX = i + ki - 1;
                    int imgY = j + kj - 1;

                    if (imgX >= 0 && imgX < m && imgY >= 0 && imgY < n) {
                        int kernelIndex = imgX * n + imgY; // specify the column index in the sparse matrix
                        tripletList.push_back(Triplet<double>(index, kernelIndex, kernel(ki, kj)));
                        count += 1;
                    }
                }
            }
        }
    }
    cout << "The number of non zero entries in the A_1 matrix are: " << count << endl;

    // Construct the sparse matrix
    A_1.setFromTriplets(tripletList.begin(), tripletList.end());
    return A_1;
}



