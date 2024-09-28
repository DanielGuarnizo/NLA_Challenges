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
    /*
    @ Note:
        char* instead of taking the string that we pass as argument 
        it takes the pointer where this string is saved in memory
     */ 
    

    // Check if all parameters are included
    if(argc < 2){
        cerr << "In programm: " << argv[0] << "is missing image path"<< endl;
        return 1;
    }

    // Load the image
    const char* input_image_path = argv[1]; // this variable is a pointer for a constant string 
    cout << "input_image_path:" << input_image_path << endl;
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
        /*
            %: given the describtion (Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white). Report the size of the matrix.)
            % we assume the the image is a gray scale image so with only one chanel 

         */

    // Check if the image was loaded correctly 
    if(!image_data){
        cerr << "Error: could not load the image: " << input_image_path << endl;
        return 1;
    }

    // if loaded then print dimentions
    cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;
}