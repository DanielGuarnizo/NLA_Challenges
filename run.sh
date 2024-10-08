
# #! COMMAND TO RUN THE MAIN FILE 
#./run.sh main ./data/images/256px-Albert_Einstein_Head.jpg

#! Comand to run for solve linear system 
# mpirun -n 4 ./test1 /home/jellyfish/shared-folder/Challenge_1_NLA/sparse_matrix.mtx /home/jellyfish/shared-folder/Challenge_1_NLA/vector.mtx sol.txt hist.txt -i cg -tol 1.0e-13
# mpirun -n 4 ./test1 matA.mtx vecB.mtx sol.txt hist.txt -i cg 

#!/bin/bash
# Check if at least one argument (the filename) was provided
if [ $# -lt 1 ]; then
    echo "Usage: ./run.sh <filename (without .cpp extension)> [args]"
    exit 1
fi

# Set the filename from the first argument
filename=$1

# Shift the positional parameters to get any additional arguments
shift

# Define directories for source code, executables, and includes
src_dir="src"
bin_dir="bin"
lib_dir="lib"
include_dir="include"

# Check if the source file exists in the src/ directory
if [ ! -f "${src_dir}/${filename}.cpp" ]; then
    echo "Error: Source file '${src_dir}/${filename}.cpp' not found."
    exit 1
fi

# Create the bin/ directory if it doesn't exist
mkdir -p "${bin_dir}"

# Compile the file using g++, saving the executable in the bin/ directory
g++ -I ${include_dir} -I ${mkEigenInc} "${src_dir}/${filename}.cpp" "${lib_dir}/image_utils.cpp" -o "${bin_dir}/${filename}"

# Check if the compilation succeeded
if [ $? -eq 0 ]; then
    echo "Compilation successful."
    
    # Run the executable with the remaining arguments
    ./"${bin_dir}/${filename}" "$@"
else
    echo "Compilation failed."
fi

