// Fundamental libraries
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <unsupported/Eigen/SparseExtra>


#include "../lib/utils.cpp"

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



MatrixXd create_checkerboard() {
    const int checkerboard_size = 200;
    const int square_size = 25;

    MatrixXd checkerboard(checkerboard_size, checkerboard_size);

    MatrixXd black_square = MatrixXd::Zero(square_size, square_size);
    MatrixXd white_square = MatrixXd::Ones(square_size, square_size);

    MatrixXd white_black(square_size, square_size*2);
    white_black.block(0, 0, square_size, square_size) = white_square;
    white_black.block(0, square_size, square_size, square_size) = black_square;

    MatrixXd black_white(square_size, square_size*2);
    black_white.block(0, 0, square_size, square_size) = black_square;
    black_white.block(0, square_size, square_size, square_size) = white_square;


    MatrixXd checkerboard_W_row(square_size, checkerboard_size);
    for (int i = 0; i < checkerboard_size; i += square_size * 2) {
        checkerboard_W_row.block(0, i, square_size, square_size * 2) = white_black;
    }

    MatrixXd checkerboard_B_row(square_size, checkerboard_size);
    for (int i = 0; i < checkerboard_size; i += square_size * 2) {
        checkerboard_B_row.block(0, i, square_size, square_size * 2) = black_white;
    }

    for (int i = 0; i < checkerboard_size; i += square_size) {
        checkerboard.block(i, 0, square_size, checkerboard_size ) = (i % 2 == 0) ? checkerboard_B_row : checkerboard_W_row;
    }

    vector<unsigned char> checkerboard_image_data(checkerboard_size * checkerboard_size);

    for (int i = 0; i < checkerboard_size; ++i) {
        for (int j = 0; j < checkerboard_size; ++j) {
            int index = (i * checkerboard_size + j);
            double pixel_value = checkerboard(i, j) * 255.0;
            checkerboard_image_data[index] = static_cast<unsigned char>(clamp(pixel_value, 0.0, 255.0));
        }
    }

    string path_checkerboard = "./data/images/checkerboard_image.png";
    saveImage(path_checkerboard, checkerboard_size, checkerboard_size, 1, checkerboard_image_data);

    return checkerboard;

}




int main(int argc, char* argv[]) {
    //! TASK 1 Compute the matrix product A_TA and report the euclidean norm of A_TA
    // Check if the image path argument is provided
    if (argc < 2) {
        cerr << "Error: Missing image path argument." << endl;
        return 1;
    }

    // Load the image
    int width, height, channels;
    unsigned char* image_data = load_image(argv[1], width, height, channels);

    MatrixXd original_image(height, width); // initialize the matrix where we will store the casting values
    
    for (int i = 0; i < height; ++i) {
        for(int j = 0; j < width; j++){
            int index = (i*width + j);
            original_image(i,j) = static_cast<double>(image_data[index]) / 255.0;
        }
    }
    // Free memory
    stbi_image_free(image_data);

    // Compute A^T * A
    MatrixXd A_transpose_A = original_image.transpose() * original_image;

    cout << "TASK 1:\n  Norm of A^T * A: " << A_transpose_A.norm() << endl;

    
    //! TASK 2: Eigenvalue computation
    // cout << " Is A_traspose A symmetric? " << A_transpose_A.isApprox(A_transpose_A.transpose()) << endl;

    // find singular values of A
    SelfAdjointEigenSolver<MatrixXd> solver(A_transpose_A);
    if (solver.info() != Success) {
        cerr << "Eigenvalue computation failed!" << endl;
        return -1;
    }

    // let's compute the square of the singular values
    VectorXd singular_values = solver.eigenvalues().cwiseSqrt();
    sort(singular_values.data(), singular_values.data() + singular_values.size(), greater<double>());

    // pick the first 2 singular values

    cout << "TASK 2:\n  First 2 singular values: " << singular_values(0) << " and " << singular_values(1) << endl;

    //! TASK 3: 

    string path = "./data/MTX_onjects/A_transpose_AA.mtx";
    saveMarket(A_transpose_A, path);

    // command that I have run
    // mpirun -n 4 ./eigen1 A_transpose_A.mtx eigvec.txt hist.txt -e si -ss 2 -etol 1.0e-8 -ie pi
    cout << "TASK 3:\n";
    cout << "  Eigenvalue 1: " << 1.608332e+04 << endl;
    cout << "  Eigenvalue 2: " << 1.394686e+03 << endl;
    
    
    //! TASK 4: Find a good shift

    cout << "TASK 4:\n";
    cout << "The best shift µ that I found that accelerated the convergence is 4.0" << endl;

    //! TASK 5: Using the SVD module of the Eigen library, perform a singular value decomposition of the matrix A.
    BDCSVD<MatrixXd> svd(original_image, ComputeThinU | ComputeThinV);
    VectorXd singular_values_A = svd.singularValues();
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();

    cout << "the norm of the matrix with singular values is: " << singular_values_A.norm() << endl;


    //! TASK 6:
    // the rank is 256

    int k_40 = 40;
    int k_80 = 80;

    MatrixXd C_40(height, k_40);
    MatrixXd D_40(width, k_40);

    MatrixXd C_80(height, k_80);
    MatrixXd D_80(width, k_80);

    DiagonalMatrix<double, Dynamic> Epsilon(singular_values_A);

    C_40 = U.leftCols(k_40);
    D_40 = (V * Epsilon.toDenseMatrix()).leftCols(k_40);

    C_80 = U.leftCols(k_80);
    D_80 = (V * Epsilon.toDenseMatrix()).leftCols(k_80);

    cout << "TASK 6:\n";
    cout << "  Number of non-zero entries of C and D with k=40:\n";
    cout << "   " << (C_40.nonZeros()) << endl;
    cout << "   " << (D_40.nonZeros()) << endl;

    cout << "  Number of non-zero entries of C and D with k=80:\n";
    cout << "   " << (C_80.nonZeros()) << endl;
    cout << "   " << (D_80.nonZeros()) << endl;

    //! TASK 7: 

    MatrixXd A_40 = C_40 * D_40.transpose();
    MatrixXd A_80 = C_80 * D_80.transpose();

    string path_40 = "./data/images/A_40_image.png";
    string path_80 = "./data/images/A_80_image.png";

    
    vector<unsigned char> output_image_data(height * width);

    for (int i = 0; i < height; ++i) {
        for(int j = 0; j < width; j++){
            int index = (i*width + j);
            double pixel_value = A_40(i,j) * 255.0;
            output_image_data[index] = static_cast<unsigned char>(clamp(pixel_value, 0.0, 255.0));
        }
    }

    saveImage(path_40, width, height, channels, output_image_data);

    for (int i = 0; i < height; ++i) {
        for(int j = 0; j < width; j++){
            int index = (i*width + j);
            double pixel_value = A_80(i,j) * 255.0;
            output_image_data[index] = static_cast<unsigned char>(clamp(pixel_value, 0.0, 255.0));
        }
    }

    saveImage(path_80, width, height, channels, output_image_data);


    //! TASK 8: Create and save a checkerboard image and report the euclidean norm
    
    MatrixXd checkerboard = create_checkerboard();

    cout << "TASK 8:\n";
    cout << "  Euclidean norm of the checkerboard image: " << checkerboard.norm() << endl;
    
    //! TASK 9: Introduce a noise into the checkerboard image by adding random fluctuations of color ranging between [−50,50] to each pixel value.

    srand(static_cast<unsigned int>(time(0)));
    MatrixXd noisy_checkerboard(checkerboard.rows(), checkerboard.cols());

    for (int i = 0; i < checkerboard.rows(); i++) {
        for (int j = 0; j < checkerboard.cols(); j++) {
            double pixel_value = static_cast<double>(checkerboard(i, j) * 255.0);
            double noised_value = (pixel_value + ((rand() % 101) - 50)) / 255.0;
            noisy_checkerboard(i, j) = std::clamp(noised_value, 0.0, 1.0);
        }
    }

    vector<unsigned char> noisy_checkerboard_image_data(checkerboard.rows() * checkerboard.cols());

    for (int i = 0; i < checkerboard.rows(); ++i) {
        for (int j = 0; j < checkerboard.cols(); ++j) {
            int index = (i * checkerboard.cols() + j);
            double pixel_value = noisy_checkerboard(i, j) * 255.0;
            noisy_checkerboard_image_data[index] = static_cast<unsigned char>(clamp(pixel_value, 0.0, 255.0));
        }
    }

    string path_noisy_checkerboard = "./data/images/noisy_checkerboard_image.png";
    saveImage(path_noisy_checkerboard, checkerboard.cols(), checkerboard.rows(), channels, noisy_checkerboard_image_data);



    //! TASK 10: perform a singular value decomposition of the matrix corresponding to the noisy image

    cout << "The noised image is symmetric: " << noisy_checkerboard.isApprox(noisy_checkerboard.transpose()) << endl; // False

    BDCSVD<MatrixXd> svd_noisy_checkerboard(noisy_checkerboard, ComputeThinU | ComputeThinV);
    VectorXd singular_values_noisy_checkerboard = svd_noisy_checkerboard.singularValues();
    MatrixXd U_noisy_checkerboard = svd_noisy_checkerboard.matrixU();
    MatrixXd V_noisy_checkerboard = svd_noisy_checkerboard.matrixV();

    sort(singular_values_noisy_checkerboard.data(), singular_values_noisy_checkerboard.data() + singular_values_noisy_checkerboard.size(), greater<double>());

    cout << "TASK 10:\n First 2 singular values of the noisy checkerboard image: " << singular_values_noisy_checkerboard(0) << " and " << singular_values_noisy_checkerboard(1) << endl;

    //! TASK 11: Compute matrices C and D

    int k_5 = 5;
    int k_10 = 10;

    DiagonalMatrix<double, Dynamic> Epsilon_noisy_checkerboard(singular_values_noisy_checkerboard);

    MatrixXd C_5 = U_noisy_checkerboard.leftCols(k_5);
    MatrixXd D_5 = (V_noisy_checkerboard * Epsilon_noisy_checkerboard.toDenseMatrix()).leftCols(k_5);

    MatrixXd C_10 = U_noisy_checkerboard.leftCols(k_10);
    MatrixXd D_10 = (V_noisy_checkerboard * Epsilon_noisy_checkerboard.toDenseMatrix()).leftCols(k_10);

    cout << "TASK 11:\n";
    cout << "Size of C and D with k=5:\n";
    cout << "  C_5: " << C_5.rows() << "x" << C_5.cols() << endl;
    cout << "  D_5: " << D_5.rows() << "x" << D_5.cols() << endl;

    cout << "Size of C and D with k=10:\n";
    cout << "  C_10: " << C_10.rows() << "x" << C_10.cols() << endl;
    cout << "  D_10: " << D_10.rows() << "x" << D_10.cols() << endl;

    //! TASK 12: Compute compressed images

    MatrixXd A_5_noisy_checkerboard = C_5 * D_5.transpose();
    MatrixXd A_10_noisy_checkerboard = C_10 * D_10.transpose();

    string path_A_5_noisy_checkerboard = "./data/images/A_5_noisy_checkerboard_image.png";
    string path_A_10_noisy_checkerboard = "./data/images/A_10_noisy_checkerboard_image.png";

    vector<unsigned char> A_5_noisy_checkerboard_image_data(checkerboard.rows() * checkerboard.cols());

    for (int i = 0; i < checkerboard.rows(); ++i) {
        for (int j = 0; j < checkerboard.cols(); ++j) {
            int index = (i * checkerboard.cols() + j);
            double pixel_value = A_5_noisy_checkerboard(i, j) * 255.0;
            A_5_noisy_checkerboard_image_data[index] = static_cast<unsigned char>(clamp(pixel_value, 0.0, 255.0));
        }
    }

    saveImage(path_A_5_noisy_checkerboard, checkerboard.cols(), checkerboard.rows(), channels, A_5_noisy_checkerboard_image_data);

    vector<unsigned char> A_10_noisy_checkerboard_image_data(checkerboard.rows() * checkerboard.cols());

    for (int i = 0; i < checkerboard.rows(); ++i) {
        for (int j = 0; j < checkerboard.cols(); ++j) {
            int index = (i * checkerboard.cols() + j);
            double pixel_value = A_10_noisy_checkerboard(i, j) * 255.0;
            A_10_noisy_checkerboard_image_data[index] = static_cast<unsigned char>(clamp(pixel_value, 0.0, 255.0));
        }
    }

    saveImage(path_A_10_noisy_checkerboard, checkerboard.cols(), checkerboard.rows(), channels, A_10_noisy_checkerboard_image_data);



    










    return 0;
}



