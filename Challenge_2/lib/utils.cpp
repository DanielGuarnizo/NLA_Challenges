// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"  // Include this only in one .cpp file
// #include "../include/utils.hpp"
// #include <iostream>
// #include <vector>

// // Libraries to create Eigen matrices 
// #include <Eigen/Dense>
// #include <Eigen/Sparse>
// using namespace Eigen;
// using namespace std;

// void saveMatrixToImage(const MatrixXd &matrix, const string &path, int n, int m, int channels){
//     // Vector to store unsigned char pixel values (for the image)
//     vector<unsigned char> output_image_data(m * n);

//     // Convert the MatrixXd to unsigned char and clamp values between 0 and 255
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             // Clamp the result to [0, 255] and cast it to unsigned char
//             output_image_data[i * n + j] = static_cast<unsigned char>(std::min(std::max((matrix(i, j) * 255.0), 0.0), 255.0));
//         }
//     }

//     // Save the image as PNG using stb_image_write
//     if (stbi_write_png(path.c_str(), n, m, channels, output_image_data.data(), n * channels) == 0) {
//         cerr << "Error: Could not save the image to " << path << endl;
//     } else {
//         //cout << "Image saved successfully to " << path << endl;
//     }
// }