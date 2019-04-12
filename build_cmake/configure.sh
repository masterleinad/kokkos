#!/bin/bash
export KOKKOS_PATH=/tmp/kokkos

mkdir -p build && cd build || exit 1
cmake -DKOKKOS_ARCH=Volta70 \
      -DKOKKOS_ENABLE_CUDA=ON \
      -DKOKKOS_ENABLE_CUDA_LAMBDA=ON \
      -DCMAKE_CXX_COMPILER=/tmp/kokkos/bin/nvcc_wrapper \
      -DCMAKE_INSTALL_PREFIX=${PWD}/install \
      ${KOKKOS_PATH}
make -j
make -j install

#need nvcc_wrapper to compile

export PKG_CONFIG_PATH=${PWD}/install/lib/pkgconfig/:${PKG_CONFIG_PATH}
echo "PKG_CONFIG_PATH: ${PKG_CONFIG_PATH}"

tree ${KOKKOS_PATH}/build_cmake/build/install &> ${KOKKOS_PATH}/build_cmake/cmake_tree
