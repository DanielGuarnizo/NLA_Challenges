#include "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/include/stb_image_write.h"
#include "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/include/stb_image.h"
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
MatrixXd H_lap; // Define H_lap here

// initialize kernels 
void initializeKernels()
{
    H_sh2 = MatrixXd::Zero(3, 3); // Initialize to zero
    H_sh2 << 0.0, -3.0, 0.0,
        -1.0, 9.0, -3.0,
        0.0, -1.0, 0.0;

    H_lap = MatrixXd::Zero(3, 3); // Initialize to zero
    H_lap << 0.0, -1.0, 0.0,
        -1.0, 4.0, -1.0,
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
                    if (imgX >= 0 && imgX < m && imgY >= 0 && imgY < n && kernel(ki, kj) != 0)
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

void loadSolutionFromFile(const string& path_filename, const int n, const int m, const int channels, const string& result_image_path) {
    ifstream infile(path_filename);
    vector<double> values;
    string line;

    // Skip the first two lines (MatrixMarket header and size)
    getline(infile, line); // Skip the header line
    getline(infile, line); // Skip the size line

    // Read each line from the MatrixMarket file (index and value)
    while (getline(infile, line)) {
        stringstream ss(line);
        int index;
        double value;

        // Extract the index (ignore) and the value
        ss >> index >> value;

        // Store the value
        values.push_back(value);
    }

    // Convert values to unsigned char and scale
    vector<unsigned char> output_image_data(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        // Clamp the value between 0.0 and 1.0, then scale to [0, 255]
        output_image_data[i] = static_cast<unsigned char>(clamp(values[i], 0.0, 255.0));
    }

    // Save the resulting image using your utility function
    saveImage(result_image_path, m, n, channels, output_image_data);
}