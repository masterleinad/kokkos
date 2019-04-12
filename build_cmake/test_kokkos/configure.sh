#!/bin/bash
mkdir -p build && cd build || exit 1
cmake -DCMAKE_CXX_COMPILER=${KOKKOS_DIR}/bin/nvcc_wrapper ..
make VERBOSE=1
