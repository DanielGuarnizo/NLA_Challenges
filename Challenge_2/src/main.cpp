
// Fundamental libraries
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// define macros to inclue STB read/write images
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

// Define namaspaces 
using namespace std;
using namespace Eigen;


int main(int argc, char* argv[]){
    // check if it has as argument the image
    if(argc < 2){
        cerr << "In program: " << argv[0] << "is missing image path of Eistein Head" << endl;
        return 1;
    }

    //! TASK 1: Load image and compute A^T*A
    const char* input_image_path = argv[1];
    int n, m, channels; 
    unsigned char* image_data = stbi_load(input_image_path, &n, &m, &channels, 1);

    // Check if the image was loaded correctly
    if(!image_data){
        cerr << "Error: could not load the image: " << input_image_path << endl;
        return 1;
    }

    // Cast image_data to a Eigen matrix A
    MatrixXd A(m,n);

    // we cast an at the same time we normalize the image 
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            int index = (i*n) + j;
            A(i,j) = static_cast<double>(image_data[index] / 255.0);
        }
    }

    // Make matrix multiplication A^T*A
    MatrixXd A_transpose_A = A.transpose() * A;

    cout << "TASK 1:\n";
    cout << "  " << A_transpose_A.norm() << endl;


    //! TASK 2: 
    


    return 0;
}