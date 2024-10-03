#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <vector>
#include <iostream>
#include <cstdlib>   // For rand() and srand()
#include <ctime>     // For time()
#include <algorithm>
#include <fstream>
#include <string>



// my utils
#include "../include/image_utils.hpp"


// Include header files to read and write in images
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"



/*
Defining the STB_IMAGE_IMPLEMENTATION and STB_IMAGE_WRITE_IMPLEMENTATION macros 
before including these headers enables the actual function implementations.
*/ 

using namespace Eigen;
using namespace std;








int main(int argc, char* argv[]){
    // Initialize the kernels before using them
    initializeKernels();
    // Seed the random number generator once, before the loop
    srand(static_cast<unsigned int>(time(0)));


   

    //! 1. POINT: LOAD IMAGE AND PRINT DIMENTIONS
    
    // Check if all parameters are included
    if(argc < 2){
        cerr << "In programm: " << argv[0] << "is missing image path"<< endl;
        return 1;
    }

    // Load the image
    int width, height, channels;
    unsigned char* image_data = load_image(argv[1], width, height, channels);


    //! 2. POINT: INTRDUCE NOISE SIGNAL INTO THE LOADED IMAGE

    MatrixXd original_image(height, width); // initialize the matrix where we will store the casting values
    
    for (int i = 0; i < height; ++i) {
        for(int j = 0; j < width; j++){
            int index = (i*width + j);
            original_image(i,j) = static_cast<int>(image_data[index]);
        }
    }
    // Free memory
    stbi_image_free(image_data);


    auto noisy_image = addNoiseToImage(original_image);


    // Save the noisy_image using stb_image_write
    const string output_image_path = "./data/images/noisy_image.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, noisy_image.data(), width) == 0){
        // c_str: is to pass the output path in C_style
        cerr << "Error: Could not save noisy image" << endl;
        return 1;
    }
    std::cout << "Image saved as " << output_image_path.substr(output_image_path.find_last_of('/') + 1) << std::endl;

    //! 3. Reshape original and noisy image to vectors v and w

    // define vector to store the matrices 
    VectorXd v(height*width);
    VectorXd w(height*width);

    // fill thse vectors
    for(int i = 0; i < height; i++){
        for(int j = 0 ; j < width; j++){
            v((i*width) + j) = static_cast<double>(original_image(i,j));
            w((i*width) + j) = static_cast<double>(noisy_image(i,j));
        }
    }

    cout << "The size of vector v is: " << v.size() << " and vector w is "<< w.size() << " where HeightxWidth is: "<< height*width << endl ;
    cout << "The Euclidean norm of v is: " << v.norm() << endl;

    //! 4. Write a convolution operation of smooth kernel H_av2 as matrix vector multiplication where A_1 is the convolutional matrix 

    // Initialize the A_1 matrix
    int m = height;
    int n = width;

    // make a function that creates the convolutional matrix and provided 
    SparseMatrix<double> A_1 = createConvolutionalMatrix(m, n, H_av2);

    //! 5 Applied the A_1 smooth filter to the noisy image doing a matrix vector multiplication

    // Multiply A_1 with w (noisy image vector)
    VectorXd result_vector = A_1 * w;

    // Vector to store unsigned char pixel values (for the image)
    vector<unsigned char> output_image_data(m * n);

    // Convert result_vector to unsigned char and clamp values between 0 and 255 in a single loop
    for (int i = 0; i < m * n; ++i) {
        // Use std::clamp to restrict values between 0 and 255, and cast to unsigned char
        output_image_data[i] = static_cast<unsigned char>(clamp(result_vector(i), 0.0, 255.0));
    }

    // Save the resulting image using utils 
    const string result_image_path = "./data/images/smooth_image.png"; //! ADD ALWAYS THE ABSOLUTE PATH, OTHERWISE IT CANNOT SAVE IT 
    saveImage(result_image_path, width, height, channels, output_image_data);



    //! 6 
    // construct A_2 matrix using the H_sh2 kernel 
    SparseMatrix<double> A_2 = createConvolutionalMatrix(n,m, H_sh2);

    // check if A_2 is symmetric
    if(A_2.isApprox(A_2.transpose()) == 0) {
        cout << "Matrix A_2 is transpose ? --> False"  << endl;
    } else {
        cout << "Matrix A_2 is transpose ? --> True"  << endl;
    }
    

    //! 7 Apply the previous sharpening filter to the original image v
    result_vector = A_2 * v;

    for (int i = 0; i < m * n; ++i) {
        // normalize
        double pixel_value = result_vector(i) / 255.0;
        // Use std::clamp to restrict values between 0 and 255, and cast to unsigned char
        output_image_data[i] = static_cast<unsigned char>(clamp(pixel_value, 0.0, 1.0)*255.0);
    }

    // Save the resulting image using utils
    const string sharpened_image_path = "./data/images/sharpened_image.png";
    saveImage(sharpened_image_path, width, height, channels, output_image_data);

    //! 8.
    // Export the Eigen matrix A2 and vector w in the .mtx format. 
    // Using a suitable iterative
    // solver and preconditioner technique available in the LIS library compute the approximate
    // solution to the linear system A2x = w prescribing a tolerance of 10−9. Report here the
    // iteration count and the final residual.


    // Export A_2 and w to the .mtx format
    string A_2_path = "./data/matrices/A_2.mtx";
    string w_path = "./data/matrices/w.mtx";

    // Export A_2 and w to the .mtx format
    Eigen::saveMarket(A_2, A_2_path);
    Eigen::saveMarketVector(w, w_path);

    return 0;
}