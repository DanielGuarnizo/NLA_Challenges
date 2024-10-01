# #!/bin/bash

# # Check if at least one argument (the filename) was provided
# if [ $# -lt 1 ]; then
#     echo "Usage: ./run.sh <filename (without .cpp extension)> [args]"
#     exit 1
# fi

# # Set the filename from the first argument
# filename=$1

# # Shift the positional parameters to get any additional arguments
# shift

# # Compile the file using g++
# g++ -I ${mkEigenInc} "${filename}.cpp" -o "${filename}"

# # Check if the compilation succeeded
# if [ $? -eq 0 ]; then
#     echo "Compilation successful."
    
#     # Run the executable with the remaining arguments
#     ./"${filename}" "$@"
# else
#     echo "Compilation failed."
# fi
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

# Define directories for source code and executables
src_dir="src"
bin_dir="bin"

# Check if the source file exists in the src/ directory
if [ ! -f "${src_dir}/${filename}.cpp" ]; then
    echo "Error: Source file '${src_dir}/${filename}.cpp' not found."
    exit 1
fi

# Create the bin/ directory if it doesn't exist
mkdir -p "${bin_dir}"

# Compile the file using g++, saving the executable in the bin/ directory
g++ -I ${mkEigenInc} "${src_dir}/${filename}.cpp" -o "${bin_dir}/${filename}"

# Check if the compilation succeeded
if [ $? -eq 0 ]; then
    echo "Compilation successful."
    
    # Run the executable with the remaining arguments
    ./"${bin_dir}/${filename}" "$@"
else
    echo "Compilation failed."
fi

#! COMMAND TO RUN THE MAIN FILE 
#./run.sh main ./data/images/256px-Albert_Einstein_Head.jpg