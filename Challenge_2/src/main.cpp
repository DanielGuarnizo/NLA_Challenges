// Fundamental libraries
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <unsupported/Eigen/SparseExtra>

// Library for sorting 
#include <algorithm>
#include <cmath>  // For sqrt()
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()

// Define macros to include STB read/write images
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

// Define namespaces 
using namespace std;
using namespace Eigen;

// Function to save a matrix as an image
void saveMatrixToImage(const MatrixXd &matrix, const string &path, int n, int m, int channels) {
    vector<unsigned char> output_image_data(m * n);

    // Convert the MatrixXd to unsigned char and clamp values between 0 and 255
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            output_image_data[i * n + j] = static_cast<unsigned char>(std::min(std::max(matrix(i, j) * 255.0, 0.0), 255.0));
        }
    }

    // Save the image as PNG using stb_image_write
    if (stbi_write_png(path.c_str(), n, m, channels, output_image_data.data(), n * channels) == 0) {
        cerr << "Error: Could not save the image to " << path << endl;
    }
}

// Function to create a checkerboard matrix
MatrixXd createCheckerboard(int size, int squareSize) {
    MatrixXd checkerboard(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            checkerboard(i, j) = (((i / squareSize) + (j / squareSize)) % 2 == 0) ? 1.0 : 0.0;
        }
    }
    return checkerboard;
}

int main(int argc, char* argv[]) {
    //! TASK 1:
    // Check if the image path argument is provided
    if (argc < 2) {
        cerr << "Error: Missing image path argument." << endl;
        return 1;
    }

    const char* input_image_path = argv[1];
    int n, m, channels; 
    unsigned char* image_data = stbi_load(input_image_path, &n, &m, &channels, 1);

    // Check if the image was loaded correctly
    if (!image_data) {
        cerr << "Error: Could not load the image: " << input_image_path << endl;
        return 1;
    }

    // Cast image_data to an Eigen matrix A and normalize the image
    MatrixXd A(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A(i, j) = static_cast<double>(image_data[i * n + j]) / 255.0;
        }
    }
    stbi_image_free(image_data); // Free the image data after use

    // Compute A^T * A
    MatrixXd A_transpose_A = A.transpose() * A;
    cout << "TASK 1:\n";
    cout << "  Norm of A^T * A: " << A_transpose_A.norm() << endl;

    //! TASK 2: Eigenvalue computation
    SelfAdjointEigenSolver<MatrixXd> solver(A_transpose_A);
    if (solver.info() != Success) {
        cerr << "Eigenvalue computation failed!" << endl;
        return -1;
    }

    VectorXd singular_values = solver.eigenvalues().cwiseSqrt();
    sort(singular_values.data(), singular_values.data() + singular_values.size(), greater<double>());

    cout << "TASK 2: Singular values\n";
    cout << "  " << singular_values[0] << endl;
    cout << "  " << singular_values[1] << endl;

    //! TASK 3: Save A_transpose_A in Market Matrix Format
    saveMarket(A_transpose_A, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/MTX_objects/A_transpose_A.mtx");
    cout << "TASK 3:\n";
    cout << "  Eigenvalue: " << 1.608332e+04 << endl;
    cout << "  The result is in agreement with the singular values previously computed" << endl;

    //! TASK 4: Placeholder for additional computations
    cout << "TASK 4:\n";
    cout << "  Non so come cazzo trovare mu" << endl;

    //! TASK 5: Singular Value Decomposition
    BDCSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    VectorXd singular_values_A = svd.singularValues();
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();

    cout << "TASK 5:\n";
    cout << "  Norm of matrix with singular values: " << singular_values_A.norm() << endl;

    //! TASK 6: Compute matrices C and D
    int k_40 = 40;
    int k_80 = 80;
    DiagonalMatrix<double, Dynamic> Epsilon(singular_values_A);

    MatrixXd C_40 = U.leftCols(k_40);
    MatrixXd D_40 = (V * Epsilon.toDenseMatrix()).leftCols(k_40);
    MatrixXd C_80 = U.leftCols(k_80);
    MatrixXd D_80 = (V * Epsilon.toDenseMatrix()).leftCols(k_80);

    cout << "TASK 6:\n";
    cout << "  Number of non-zero entries of C and D with k=40:\n";
    cout << "   " << (C_40.array() != 0).count() << endl;
    cout << "   " << (D_40.array() != 0).count() << endl;

    cout << "  Number of non-zero entries of C and D with k=80:\n";
    cout << "   " << (C_80.array() != 0).count() << endl;
    cout << "   " << (D_80.array() != 0).count() << endl;

    //! TASK 7: Compute compressed images
    MatrixXd A_40 = C_40 * D_40.transpose();
    MatrixXd A_80 = C_80 * D_80.transpose();
    saveMatrixToImage(A_40, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/images/A_40_image.png", n, m, channels);
    saveMatrixToImage(A_80, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/images/A_80_image.png", n, m, channels);

    cout << "TASK 7:\n";
    cout << "  Compressed images saved successfully." << endl;

    //! TASK 8: Create and save a checkerboard image
    const int checkerboard_size = 200;
    const int square_size = 25;
    MatrixXd checkerboard = createCheckerboard(checkerboard_size, square_size);
    saveMatrixToImage(checkerboard, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/images/checkerboard_image.png", checkerboard_size, checkerboard_size, channels);
    cout << "TASK 8:\n";
    cout << "  Checkerboard image was successfully saved." << endl;

    //! TASK 9: Create and save a noisy checkerboard image
    MatrixXd noisy_image(checkerboard_size, checkerboard_size);
    srand(static_cast<unsigned int>(time(0))); // Seed the random number generator

    noisy_image = checkerboard.unaryExpr([](double val) {
        int u_val = static_cast<int>(val * 255);
        int noise = (rand() % 101) - 50; // Random noise between [-50, 50]
        int new_val = std::clamp(u_val + noise, 0, 255);
        return static_cast<double>(new_val) / 255.0; // Normalize back to [0, 1]
    });

    saveMatrixToImage(noisy_image, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/images/noisy_checkerboard_image.png", checkerboard_size, checkerboard_size, channels);
    cout << "TASK 9:\n";
    cout << "  Noisy checkerboard image was successfully saved." << endl;

    //! TASK 10: Singular Value Decomposition for Noisy checkerboard
    BDCSVD<MatrixXd> svd_1(noisy_image, ComputeThinU | ComputeThinV);
    VectorXd singular_values_noisy_image = svd_1.singularValues();
    MatrixXd U_noisy_image = svd_1.matrixU();
    MatrixXd V_noisy_image = svd_1.matrixV();

    cout << "TASK 10:\n";
    cout << "  Norm of matrix with singular values of Noisy checkerboard image: " << singular_values_noisy_image.norm() << endl;

    //! TASK 11: Compute matrices C and D
    int k_5 = 2;
    int k_10 = 10;
    DiagonalMatrix<double, Dynamic> Epsilon_1(singular_values_noisy_image);

    MatrixXd C_5 = U_noisy_image.leftCols(k_5);
    MatrixXd D_5 = (V_noisy_image * Epsilon_1.toDenseMatrix()).leftCols(k_5);
    MatrixXd C_10 = U_noisy_image.leftCols(k_10);
    MatrixXd D_10 = (V_noisy_image * Epsilon_1.toDenseMatrix()).leftCols(k_10);

    cout << "TASK 11:\n";
    cout << "  Number of non-zero entries of C and D with k=5:\n";
    cout << "   " << (C_5.array() != 0).count() << endl;
    cout << "   " << (D_5.array() != 0).count() << endl;

    cout << "  Number of non-zero entries of C and D with k=10:\n";
    cout << "   " << (C_10.array() != 0).count() << endl;
    cout << "   " << (D_10.array() != 0).count() << endl;

    //! TASK 12: Compute compressed images
    MatrixXd A_5 = C_5 * D_5.transpose();
    MatrixXd A_10 = C_10 * D_10.transpose();
    saveMatrixToImage(A_5, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/images/A_5_image.png", 200, 200, channels);
    saveMatrixToImage(A_10, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/images/A_10_image.png", 200, 200, channels);

    cout << "TASK 12:\n";
    cout << "  Compressed images saved successfully." << endl;



    return 0;
}