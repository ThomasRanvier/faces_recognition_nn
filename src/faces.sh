#!/bin/bash

# Script for executing the Python version of Faces
# Usage:
# bash faces.sh <training_file> <facit_file> <test_file>

# the location of this script:
base_dir="$(dirname "$0")"

python2 $base_dir/faces.py $1 $2 $3

