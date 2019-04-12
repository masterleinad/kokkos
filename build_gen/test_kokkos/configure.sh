#!/bin/bash
KOKKOS_DIR=/tmp/kokkos/build_gen/build/install
mkdir -p build && cd build || exit 1
cmake -DCMAKE_CXX_COMPILER=${KOKKOS_DIR}/bin/nvcc_wrapper -DKOKKOS_DIR=${KOKKOS_DIR} ..
make -j
