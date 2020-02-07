ARG BASE=nvidia/cuda:10.1-devel
FROM $BASE

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get upgrade -y && apt-get install -y \
      bc \
      cmake \
      git \
      wget \
      ccache \
      libelf-dev \
      && \
      apt-get clean && rm -rf /var/lib/apt/lists/*

# Install HIP
ARG HIP_VERSION=roc-2.6.0
ENV PATH=/opt/rocm/hip/bin:$PATH
RUN HIP_URL=https://github.com/ROCm-Developer-Tools/HIP/archive/${HIP_VERSION}.tar.gz && \
    mkdir /scratch && cd /scratch && \
    wget --quiet ${HIP_URL} --output-document=HIP.tar.gz && \
    mkdir HIP && tar -xf HIP.tar.gz -C HIP --strip-components=1 && \
    cd HIP && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && make && make install && \
    rm -r /scratch && \
    hipconfig --full
