#!/bin/bash
#cp -r ../ccg_lstm_tagger .
#cp ../dynet/* dynet/
rm CMakeCache.txt
rm -rf CMakeFiles
#cmake . -DEIGEN3_INCLUDE_DIR=/home/tzy/1.deeplibs/eigen_2017_03_05/eigen_libs/
#cmake . -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3 -DMKL_ROOT="/opt/intel/mkl/"
cmake . -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3
make -j 8

