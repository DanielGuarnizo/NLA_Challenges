#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>   // For rand() and srand()
#include <ctime>     // For time()

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
    //@ Note:char* instead of taking the string that we pass as argument 
    //@ it takes the pointer where this string is saved in memory
   

    //! 1. POINT: LOAD IMAGE AND PRINT DIMENTIONS
    
    // Check if all parameters are included
    if(argc < 2){
        cerr << "In programm: " << argv[0] << "is missing image path"<< endl;
        return 1;
    }

    // Load the image
    const char* input_image_path = argv[1]; // this variable is a pointer for a constant string, so a memory adress where the image is stored
    cout << "input_image_path:" << input_image_path << endl;
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
        //% It decodes the image data into a linear array of pixels, stored in image_data, where each pixel’s color is represented by one or more bytes (depending on the number of channels).
        //% image_data[0]: refers to the first byte of the loaded image data, if gray scale image then if the first color, if RGB image then it's the first channel.
        //% image_data[0]: accesses the first pixel value (the first byte of the image data), not the first character of the pointer’s value (the address).
    
    //@ TESTS
    // cout << "Firts pixel value without static_cast: " << image_data[0] << endl; // here i am seeing the asci code that represent the 64 value of the first index
    // cout << "First pixel value with static_cast: " << static_cast<int>(image_data[0]) << endl;
    // cout << "Pointer address: " << static_cast<void*>(image_data) << endl;

    // Check if the image was loaded correctly 
    if(!image_data){
        cerr << "Error: could not load the image: " << input_image_path << endl;
        return 1;
    }

    // if loaded then print dimentions
    cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;

    //! 2. POINT: INTRDUCE NOISE SIGNAL INTO THE LOADED IMAGE

    // First we cast each value of the image to an int value, in such a way to be able to perform operations
    MatrixXd original_image(height, width); // initialize the matrix where we will store the casting values
    
    for (int i = 0; i < height; ++i) {
        for(int j = 0; j < width; j++){
            int index = (i*width + j);
            original_image(i,j) = static_cast<int>(image_data[index]);
        }
    }
    // Free memory
    stbi_image_free(image_data);

    // check top left corner of the original image
    cout << "ORIGINAL IMAGE:" << endl << original_image.topLeftCorner(6,6) << endl;

    // first we have to define the noise image 
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noisy_image(height, width);
        //% We have to define the matrix in this way because we don't knwo yet the size of the matrix
        //% then the size will be found in run time, also we want to be sure it's unsigned char, given that we know for sure 
        //% that we are handle positive values in a range of [0,255], given that is a gray image 

    // Seed the random number generator once, before the loop
    srand(static_cast<unsigned int>(time(0)));

    // We will applied to each pixel a random fluctation of color ranging between [-50,50]
    noisy_image = original_image.unaryExpr([](int val) -> unsigned char {
        // Generate random noise in range [-50, 50]
        int noise = (rand() % 101) - 50; // random number between 0 and 100, then shift to [-50, 50]

        // Apply noise, ensuring values stay within the [0, 255] range
        int new_val = val + noise;
        if (new_val < 0) new_val = 0;
        if (new_val > 255) new_val = 255;
        return static_cast<unsigned char>(new_val);
    });

    // Save the noisy_image using stb_image_write
    const string output_image_path = "/home/jellyfish/shared-folder/Challenge_1_NLA/data/images/noisy_image.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, noisy_image.data(), width) == 0){
        // c_str: is to pass the output path in C_style
        cerr << "Error: Could not save noisy image" << endl;
        return 1;
    }

    cout << "Noisy image saved to " << output_image_path << endl;

    //! 3. Reshape original and noisy image to vectors v and w

    // define vector to store the matrices 
    VectorXd v(height*width);
    VectorXd w(height*width);

    // create a Dynamic vector 
    // MatrixXd<double, Dynamic, 1> v(height*width);
    // MatrixXd<double, Dynamic, 1> w(height*width);

    // Matrix<int, Dynamic, 1> v(5);

    // fill thse vectors
    for(int i = 0; i < height; i++){
        for(int j = 0 ; j < width; j++){
            v((i*width) + j) = static_cast<double>(original_image(i,j));
            w((i*width) + j) = static_cast<double>(noisy_image(i,j));
        }
    }

    cout << "The size of vector v is: " << v.size() << " and vector w is "<< w.size() << " where HeightxWidth is: "<< height*width << endl ;

    //! 4. Write a convolution operation of smooth kernel H_av2 as matrix vector multiplication where A_1 is the convolutional matrix 

    // Initialize the A_1 matrix
    int m = height;
    int n = width;
    MatrixXd A_1 = Zero(m*n, m*n);

    // initialise the kernel H_av2
    double constantValue = (1/9);
    Matrix H_av2 = Constant(3,3, constantValue);
    int km = H_av2.size();
    int kn = H_av2[0].size();

    // Fill the A_1 matrix, in such a way to create the convolutional matrix over a vector image
    int non_zero = 0, // this count the number of non-zero entris that has the A_1 matrix

    // make iteration over the A_1 matrix
    for(int i=0; i < m; i++){
        for(int j=0; j< n; j++){
            int index = (i*n) + j;
            // make iteration over kernel matrix
            for(int ki = 0; ki < km; ki++){
                for(int kj = 0; kj < km; kj++){
                    // offsets
                    int x_offset = 
                    // check if the pixel is inside or outside
                    if()
                }
            }
        }
    }









    return 0;
}