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

# Compile the file using g++
g++ -I ${mkEigenInc} "${filename}.cpp" -o "${filename}"

# Check if the compilation succeeded
if [ $? -eq 0 ]; then
    echo "Compilation successful."
    
    # Run the executable with the remaining arguments
    ./"${filename}" "$@"
else
    echo "Compilation failed."
fi