#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cstdlib>   // For rand() and srand()
#include <ctime>     // For time()
#include <fstream>  // library to read files in c++ 
#include <unsupported/Eigen/SparseExtra>

// iterative solvers 
#include <Eigen/SparseLU>  // Needed for IncompleteLU
#include <Eigen/IterativeLinearSolvers>  // Needed for BiCGSTAB

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
// Function to export an Eigen sparse matrix to .mtx format
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
        cout << "  Sparse matrix exported to " << filename << "\n";
    } else {
        cerr << "  Unable to open file for writing matrix.\n";
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
        cout << "  Vector exported to " << filename << "\n";
    } else {
        cerr << "  Unable to open file for writing vector.\n";
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
        output_image_data[i] = static_cast<unsigned char>(std::min(std::max((values[i] * 255.0), 0.0), 255.0));
    }

    // Save the resulting image using your utility function
    saveImage(result_image_path, n, m, channels, output_image_data);
}

int main(int argc, char* argv[]){
    // Initialize the kernels before using them
    initializeKernels();

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
    cout << "TASK 1:\n" << "  " <<height << "x" << width << endl;

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
    // cout << "ORIGINAL IMAGE:" << endl << original_image.topLeftCorner(6,6) << endl;

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
    const string output_image_path = "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/images/noisy_image.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, noisy_image.data(), width) == 0){
        // c_str: is to pass the output path in C_style
        cerr << "Error: Could not save noisy image" << endl;
        return 1;
    }


    cout << "TAKS 2:\n"<< "  " << "Noisy image saved to "<< output_image_path << endl;

    //! 3. Reshape original and noisy image to vectors v and w and perform norm 

    // define vector to store the matrices 
    VectorXd v(height*width);
    VectorXd w(height*width);

    // fill thse vectors and normalize them 
    for(int i = 0; i < height; i++){
        for(int j = 0 ; j < width; j++){
            v((i*width) + j) = static_cast<double>(original_image(i,j) / 255.0);
            w((i*width) + j) = static_cast<double>(noisy_image(i,j) / 255.0);
        }
    }

    cout << "TASK 3: \n";
    cout << "  The size of vector v is: " << v.size() << " and vector w is "<< w.size() << " where HeightxWidth is: "<< height*width << endl ;
    cout << "  " << v.norm() << endl;
    //cout << "  " << original_image.norm() << endl;

    //! 4. Write a convolution operation of smooth kernel H_av2 as matrix vector multiplication where A_1 is the convolutional matrix 

    // Initialize the A_1 matrix
    int m = height;
    int n = width;

    // make a function that creates the convolutional matrix and provided 
    cout << "TASK 4:\n";
    SparseMatrix<double> A_1 = createConvolutionalMatrix(m, n, H_av2, "1");

    //! 5 Applied the A_1 smooth filter to the noisy image doing a matrix vector multiplication

    // Multiply A_1 with w (noisy image vector)
    cout << "TASK 5:\n";
    const string smooth_image_path = "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/images/smooth_image.png"; //! ADD ALWAYS THE ABSOLUTE PATH, OTHERWISE IT CANNOT SAVE IT 
    appliedConvolutionToImage(A_1, w, smooth_image_path, n, m, channels);
        //% in this fucntion the matrix vector multiplication is computed and then the image is saved
        //% using a function saveImage() to finish the procedure 

    cout << "  Smooth image saved into " << smooth_image_path << endl;

    //! 6 Write a convolutional operator for the sharpeing kernel H_sh2
    cout << "TASK 6:\n";
    // construct A_2 matrix using the H_sh2 kernel 
    SparseMatrix<double> A_2 = createConvolutionalMatrix(m,n, H_sh2, "2");

    // check if A_2 is symmetric
    if(A_2.isApprox(A_2.transpose()) == 0) {
        cout << "  Matrix A_2 is symmetric ? --> False"  << endl;
    } else {
        cout << "  Matrix A_2 is symmetric ? --> True"  << endl;
    }
        //% if the matrix is not symmetric to solve a linear system we have to use a iterative methods suitable for not symmetric matrices 
    
    //! 7 Applied previous convolution to the original image

    cout << "TASK 7:\n";
    // Multiply A_2 with v (original image vector)
    const string sharpen_image_path = "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/images/sharpen_image.png"; //! ADD ALWAYS THE ABSOLUTE PATH, OTHERWISE IT CANNOT SAVE IT 
    appliedConvolutionToImage(A_2, v, sharpen_image_path, n, m, channels);
        //% in this fucntion the matrix vector multiplication is computed and then the image is saved
        //% using a function saveImage() to finish the procedure
    cout << "  Sharpen image saved into " << sharpen_image_path << endl;

    //! 8 Export A_2 and w in .mtx format and compute the approximate solution using the LIS library
     
    // Using provided functions 
    saveMarket(A_2, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/MTX_objects/A_2.mtx");
    saveMarketVector(w, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/MTX_objects/w.mtx");

    // using my functions 
    exportSparseMatrixToMTX(A_2, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/MTX_objects/my_A_2.mtx");
    exportVectorToMTX(w, "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/MTX_objects/my_w.mtx");

    
    //! 9 Import the solution on the previous iteration and save it as a png image 
    const string path_filename = "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/MTX_objects/sol_x.txt";
    const string approximate_solution_x_image_path = "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/images/approximate_solution_x_image.png"; 

    loadSolutionFromFile(path_filename, n, m, channels, approximate_solution_x_image_path);

    cout << "TASK 9:\n";
    cout << "  Approximate x solution saved correctly into " << approximate_solution_x_image_path <<endl;

    //! 10 
    cout << "TASK 10:\n";
    // construct A_3 matrix using the H_lap kernel 
    SparseMatrix<double> A_3 = createConvolutionalMatrix(m,n, H_lap, "3");

    // check if A_3 is symmetric
    if(A_3.isApprox(A_3.transpose()) == 0) {
        cout <<  "  Matrix A_3 is symmetric ? --> False"  << endl;
    } else {
        cout << "  Matrix A_3 is symmetric ? --> True"  << endl;
    }

    //! 11  Matrix vector multiplication with the previous sharpen matrix  
    // Multiply A_3 with v (original image vector)
    const string edge_image_path = "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/images/edge_image.png"; //! ADD ALWAYS THE ABSOLUTE PATH, OTHERWISE IT CANNOT SAVE IT 
    appliedConvolutionToImage(A_3, v, edge_image_path, n, m, channels);
    cout << "TASK 11:\n";
    cout << "  Edge image saved into " << edge_image_path << endl;

    //! 12  compute the approximate solution to the linear system (I+A3)*y = w
    cout << "TASK 12:\n";

    SparseMatrix<double> I(m * n, m * n);
    I.setIdentity();

    // Ensure that I_minus_A3 is defined before use
    SparseMatrix<double> I_minus_A3 = I + A_3; // Subtract A_3 from identity matrix I

    VectorXd y(I_minus_A3.rows());
    VectorXd b = w; // Assume you're solving I_minus_A3 * y = v, so b = v

    // Set parameters for solver
    double tol = 10.e-10; // Convergence tolerance (unchanged)
    int maxit = 2000;     // Maximum iterations

    // Create a preconditioner, Incomplete Cholesky is suitable for symmetric matrices
    Eigen::IncompleteCholesky<double> ichol(I_minus_A3);

    // Set up and use Conjugate Gradient solver
    Eigen::ConjugateGradient<SparseMatrix<double>, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<double>> cg;
    cg.setMaxIterations(maxit);
    cg.setTolerance(tol);
    cg.compute(I_minus_A3);

    y = cg.solve(b);

    // Output solver results
    std::cout << "  Number of iterations:     " << cg.iterations() << std::endl;
    std::cout << "  Relative residual: " << cg.error() << std::endl;      
            

    // save y vector as image 
    vector<unsigned char> output_image_y(m * n);

    // Convert the MatrixXd to unsigned char and clamp values between 0 and 255
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // Clamp the result to [0, 255] and cast it to unsigned char
            output_image_y[i * n + j] = static_cast<unsigned char>(std::min(std::max((y(i * n + j) * 255.0), 0.0), 255.0));
        }
    }

    // Save the resulting image using utils 
    const string approximate_solutionEigen_y_image = "/home/jellyfish/shared-folder/NLA_Challenges/Challenge_1/data/images/approximate_solutionEigen_y_image.pngg"; 
    saveImage(approximate_solutionEigen_y_image, n, m, channels, output_image_y);

    return 0;
    
}



 