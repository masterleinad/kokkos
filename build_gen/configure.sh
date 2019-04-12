#!/bin/bash
export KOKKOS_PATH=/tmp/kokkos

mkdir build && cd build || exit 1
${KOKKOS_PATH}/generate_makefile.bash --arch=Volta70 --with-cuda

make -j
