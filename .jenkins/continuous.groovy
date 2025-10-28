pipeline {
    agent none

    environment {
        CCACHE_DIR = '/tmp/ccache'
        CCACHE_MAXSIZE = '5G'
        CCACHE_CPP2 = 'true'
        GTEST_SHUFFLE = 1
    }

    options {
        disableConcurrentBuilds(abortPrevious: true)
        timeout(time: 6, unit: 'HOURS')
    }

    triggers {
        issueCommentTrigger('.*test this please.*')
    }

    stages {
        stage('Build-1') {
            parallel {
                stage('CUDA-12.2-NVCC-RDC') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.nvcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04@sha256:5f603101462baa721ff6ddc44af82f6e9ba7cbd92a424c9f9f348e6e9d6d64c3 --build-arg ADDITIONAL_PACKAGES="gfortran clang" --build-arg CMAKE_VERSION=3.25.3'
                            label 'nvidia-docker && (volta || ampere)'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES --env NODE_NAME=${env.NODE_NAME} --env STAGE_NAME=${env.STAGE_NAME}'
                        }
                    }
                    environment {
                        OMP_NUM_THREADS = 8
                        // Nested OpenMP does not work for this configuration,
                        // so disabling it
                        OMP_MAX_ACTIVE_LEVELS = 1
                        OMP_PLACES = 'threads'
                        OMP_PROC_BIND = 'spread'
                        NVCC_WRAPPER_DEFAULT_COMPILER = 'g++-11'
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''#!/bin/bash
                              exec > >(awk '{ print "[" ENVIRON["STAGE_NAME"] "]", $0 }') 2>&1 && \
                              echo "Hostname: ${NODE_NAME}" && \
                              rm -rf install && mkdir -p install && \
                              rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=Release \
                                -DCMAKE_CXX_COMPILER=g++-11 \
                                -DCMAKE_CXX_FLAGS=-Werror \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DCMAKE_VERBOSE_MAKEFILE=ON \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_OPENMP=OFF \
                                -DKokkos_ENABLE_CUDA=ON \
                                -DKokkos_ENABLE_CUDA_LAMBDA=OFF \
                                -DKokkos_ENABLE_CUDA_UVM=ON \
                                -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                                -DKokkos_ENABLE_MULTIPLE_CMAKE_LANGUAGES=ON \
                                \
                                -DCMAKE_INSTALL_PREFIX=${PWD}/../install \
                              .. && \
                              make -j8 install && \
                              cd .. && \
                              rm -rf build-tests && mkdir -p build-tests && cd build-tests && \
                              export CMAKE_PREFIX_PATH=${PWD}/../install && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=Release \
                                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                -DCMAKE_CXX_COMPILER=$WORKSPACE/bin/nvcc_wrapper \
                                -DCMAKE_CXX_FLAGS="-Werror --Werror=all-warnings -Xcudafe --diag_suppress=940" \
                                -DCMAKE_EXE_LINKER_FLAGS="-Xnvlink -suppress-stack-size-warning" \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DCMAKE_VERBOSE_MAKEFILE=ON \
                                -DKokkos_INSTALL_TESTING=ON \
                              .. && \
                              make -j8 && ctest --no-compress-output -T Test --verbose && \
                              cd ../example/build_cmake_installed && \
                              rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_CXX_COMPILER=g++-11 \
                                -DCMAKE_CXX_FLAGS=-Werror \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DCMAKE_VERBOSE_MAKEFILE=ON \
                              .. && \
                              make -j8 && ctest --verbose && \
                              cd ../../build_cmake_installed_multilanguage && \
                              rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_CXX_COMPILER=g++-11 \
                                -DCMAKE_CXX_FLAGS=-Werror \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DCMAKE_VERBOSE_MAKEFILE=ON \
                              .. && \
                              make -j8 && ctest --verbose && \
                              cd ../.. && \
                              cmake \
                                -B build_cmake_installed_different_compiler/build \
                                -DCMAKE_CXX_COMPILER=clang++ \
                                -DCMAKE_CXX_FLAGS=-Werror \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DCMAKE_VERBOSE_MAKEFILE=ON \
                                build_cmake_installed_different_compiler && \
                              cmake --build build_cmake_installed_different_compiler/build --target all && \
                              cmake --build build_cmake_installed_different_compiler/build --target test'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build-tests/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
            }
        }
    }
}
