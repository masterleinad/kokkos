#!/bin/bash
mkdir -p build && cd build || exit 1
cmake -DCMAKE_CXX_COMPILER=clang++-7 ..
make -j
