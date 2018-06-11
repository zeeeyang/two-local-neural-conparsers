#!/bin/bash
rm CMakeCache.txt
rm -rf CMakeFiles
cmake . -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3 -DBACKEND=cuda
make -j 8

