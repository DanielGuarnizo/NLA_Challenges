#include "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/include/stb_image_write.h"
#include "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/include/stb_image.h"
#include "../include/utils.hpp"
#include <iostream>
#include <vector>

#include <fstream>

// libraries to create Eigen matrices
#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;
using namespace std;



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


void exportSparseMatrixToMTX(const SparseMatrix<double>& A, const std::string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        // Write the Matrix Market header for a sparse matrix
        file << "%%MatrixMarket matrix coordinate real general\n";
        file << A.rows() << " " << A.cols() << " " << A.nonZeros() << "\n";
        // Iterate over non-zero elements in the sparse matrix
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
                file << it.row() + 1 << " " << it.col() + 1 << " " << it.value() << "\n";  // 1-based indexing
            }
        }
        file.close();
        cout << "Sparse matrix exported to " << filename << "\n";
    } else {
        cerr << "Unable to open file for writing matrix.\n";
    }
}

// Function to export an Eigen vector to .mtx format
void exportVectorToMTX(const VectorXd& w, const std::string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        // Write the Matrix Market header for an array (vector)
        file << "%%MatrixMarket vector coordinate real general\n";
        file << w.size() << "\n";  // Vector is treated as a single column matrix
        // Write the vector elements
        for (int i = 0; i < w.size(); ++i) {
            file << i + 1 << " " << w(i) << "\n";
        }
        file.close();
        cout << "Vector exported to " << filename << "\n";
    } else {
        cerr << "Unable to open file for writing vector.\n";
    }
}