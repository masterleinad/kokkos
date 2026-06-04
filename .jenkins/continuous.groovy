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
                stage('spack-cuda') {
                    agent {
                        docker {
                          image 'nvidia/cuda:12.9.0-devel-ubuntu24.04'
                          label 'nvidia-docker && ampere'
                          registryCredentialsId 'dockerhub'
                        }
                    }
                    steps {
                        sh '''
                          DEBIAN_FRONTEND=noninteractive && \
                          apt-get update && apt-get upgrade -y && apt-get install -y \
                          build-essential \
                          wget \
                          git \
                          bc \
                          libxml2 \
                          python3-dev \
                          gfortran \
                          && \
                          apt-get clean && rm -rf /var/lib/apt/lists/*

                          export CDASH_ARGS="${SPACK_CDASH_ARGS} --cdash-build=spack-cuda"
                          rm -rf spack && \
                          git clone https://github.com/spack/spack.git && \
                          . ./spack/share/spack/setup-env.sh && \
                          spack install -v --only=dependencies kokkos@develop+cuda+wrapper+tests cuda_arch=80 ^cuda@12.9.0 && \
                          spack install -v --only=package ${CDASH_ARGS} kokkos@develop+cuda+wrapper+tests cuda_arch=80 ^cuda@12.9.0 && \
                          spack load cmake  && \
                          spack load kokkos-nvcc-wrapper && \
                          spack load cuda && \
                          spack load kokkos && \
                          spack test run ${CDASH_ARGS} kokkos && \
                          spack test results -l
                          '''
                    }
                }
            }
        }
    }
}
