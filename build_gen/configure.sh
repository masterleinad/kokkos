#!/bin/bash
KOKKOS_PATH=/tmp/kokkos

mkdir -p build && cd build || exit 1
${KOKKOS_PATH}/generate_makefile.bash \
 --arch=Volta70 \
 --with-cuda \
 --with-openmp \
 --with-cuda-options=enable_lambda \
 --compiler=/tmp/kokkos/bin/nvcc_wrapper

# missing --expt-extended-lambda if not nvcc_wrapper

make -j
make -j install

export PKG_CONFIG_PATH=${PWD}/install/lib/pkgconfig/:${PKG_CONFIG_PATH}
echo "PKG_CONFIG_PATH: ${PKG_CONFIG_PATH}"

tree ${KOKKOS_PATH}/build_gen/build/install &> ${KOKKOS_PATH}/build_gen/gen_tree
