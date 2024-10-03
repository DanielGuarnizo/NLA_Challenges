#include "/home/jellyfish/shared-folder/Challenge_1_NLA/include/stb_image_write.h"
#include "/home/jellyfish/shared-folder/Challenge_1_NLA/include/stb_image.h"
#include "../include/image_utils.hpp"
#include <iostream>
#include <vector>

#include <fstream>

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
void initializeKernels()
{
    H_sh2 = MatrixXd::Zero(3, 3); // Initialize to zero
    H_sh2 << 0.0, -3.0, 0.0,
        -1.0, 9.0, -3.0,
        0.0, -1.0, 0.0;
}

// Initialize functions

unsigned char *load_image(const char *input_image_path, int &width, int &height, int &channels)
{
    std::cout << "input_image_path:" << input_image_path << std::endl;
    unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data)
    {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        exit(1);
    }

    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;
    return image_data;
}

void saveImage(const string &path, int width, int height, int channels, const vector<unsigned char> &image_data)
{
    // Save the image as PNG using stb_image_write
    if (stbi_write_png(path.c_str(), width, height, channels, image_data.data(), width * channels) == 0)
    {
        std::cerr << "Error: Could not save the image to " << path << std::endl;
    }
    else
    {
        // print that the image was saved using the name of the image and not the path
        std::cout << "Image saved as " << path.substr(path.find_last_of('/') + 1) << std::endl;
    }
}

// Function to add noise to the image
Matrix<unsigned char, Dynamic, Dynamic, RowMajor> addNoiseToImage(const MatrixXd &original_image)
{
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noisy_image(original_image.rows(), original_image.cols());

    noisy_image = original_image.unaryExpr([](int val) -> unsigned char
                                           {
        // Generate random noise in range [-50, 50]
        double pixel_value = static_cast<double>(val); // The val is the pixel value between 0 and 255
        double noised_value = (pixel_value + ((rand() % 101) - 50)) / 255.0;
        noised_value = std::clamp(noised_value, 0.0, 1.0);
        return static_cast<unsigned char>(noised_value * 255.0); });

    return noisy_image;
}

SparseMatrix<double> createConvolutionalMatrix(const int m, const int n, const MatrixXd &kernel){
    // Initialize the A_1 sparse matrix
    SparseMatrix<double> A_1(m * n, m * n);

    // Reserve space for the triplets (since kernel size is fixed, this estimate works)
    std::vector<Triplet<double>> tripletList;
    tripletList.reserve(m * n * kernel.size());

    // Fill sparse matrix A_1
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int index = (i * n) + j; // specify the row index in the sparse matrix
            for (int ki = 0; ki < 3; ki++)
            {
                for (int kj = 0; kj < 3; kj++)
                {
                    int imgX = i + ki - 1;
                    int imgY = j + kj - 1;

                    // Check if the indices are within bounds
                    if (imgX >= 0 && imgX < m && imgY >= 0 && imgY < n)
                    {
                        int kernelIndex = imgX * n + imgY;                            // specify the column index in the sparse matrix
                        tripletList.emplace_back(index, kernelIndex, kernel(ki, kj)); // Use emplace_back for efficiency
                    }
                }
            }
        }
    }

    // Construct the sparse matrix
    A_1.setFromTriplets(tripletList.begin(), tripletList.end());

    // Output the number of non-zero entries directly using tripletList size
    std::cout << "The number of non-zero entries in the A_1 matrix are: " << tripletList.size() << std::endl;

    return A_1;
}