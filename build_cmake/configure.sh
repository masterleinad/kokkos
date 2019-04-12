#!/bin/bash
export KOKKOS_PATH=/tmp/kokkos

mkdir -p build && cd build || exit 1
cmake -DKOKKOS_ARCH=Volta70 \
      -DKOKKOS_ENABLE_CUDA=ON \
      -DCMAKE_CXX_COMPILER=/tmp/kokkos/bin/nvcc_wrapper \
      -DCMAKE_INSTALL_PREFIX=${PWD}/install \
      ${KOKKOS_PATH}
make -j
make -j install
