#include <Eigen/Dense>
#include <iostream>

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
    MatrixXd original_image(width, height); // initialize the matrix where we will store the casting values
    
    for (int i = 0; i < height; ++i) {
        for(int j = 0; j < width; j++){
            int index = (i*width + j);
            original_image(i,j) = static_cast<int>(image_data[index]);
        }
    }
    // Free memory
    stbi_image_free(image_data);

    cout << "ORIGINAL IMAGE:" << endl << original_image.topLeftCorner(6,6),


    // We will applied to each pixel a random fluctation of color ranging between [-50,50]

    


}