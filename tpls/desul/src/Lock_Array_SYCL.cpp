/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#include <cinttypes>
#include <desul/atomics/Lock_Array.hpp>
#include <sstream>
#include <string>

#ifdef DESUL_HAVE_SYCL_ATOMICS
#ifdef DESUL_SYCL_RDC
namespace desul {
namespace Impl {
__device__ __constant__ int32_t* SYCL_SPACE_ATOMIC_LOCKS_DEVICE = nullptr;
__device__ __constant__ int32_t* SYCL_SPACE_ATOMIC_LOCKS_NODE = nullptr;
}  // namespace Impl
}  // namespace desul
#endif

namespace desul {

namespace {

__global__ void init_lock_arrays_sycl_kernel() {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < SYCL_SPACE_ATOMIC_MASK + 1) {
    Impl::SYCL_SPACE_ATOMIC_LOCKS_DEVICE[i] = 0;
    Impl::SYCL_SPACE_ATOMIC_LOCKS_NODE[i] = 0;
  }
}

}  // namespace

namespace Impl {

int32_t* SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
int32_t* SYCL_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;

// Putting this into anonymous namespace so we don't have multiple defined symbols
// When linking in more than one copy of the object file
namespace {

void check_error_and_throw_sycl(syclError_t e, const std::string msg) {
  if (e != syclSuccess) {
    std::ostringstream out;
    out << "Desul::Error: " << msg << " error(" << syclGetErrorName(e)
        << "): " << syclGetErrorString(e);
    throw std::runtime_error(out.str());
  }
}

}  // namespace

template <typename T>
void init_lock_arrays_sycl() {
  if (SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h != nullptr) return;

  auto error_malloc1 = syclMalloc(&SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h,
                                 sizeof(int32_t) * (SYCL_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_sycl(error_malloc1,
                            "init_lock_arrays_sycl: syclMalloc device locks");

  auto error_malloc2 = syclHostMalloc(&SYCL_SPACE_ATOMIC_LOCKS_NODE_h,
                                     sizeof(int32_t) * (SYCL_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_sycl(error_malloc2,
                            "init_lock_arrays_sycl: syclMallocHost host locks");

  auto error_sync1 = syclDeviceSynchronize();
  DESUL_IMPL_COPY_SYCL_LOCK_ARRAYS_TO_DEVICE();
  check_error_and_throw_sycl(error_sync1, "init_lock_arrays_sycl: post malloc");

  init_lock_arrays_sycl_kernel<<<(SYCL_SPACE_ATOMIC_MASK + 1 + 255) / 256, 256>>>();

  auto error_sync2 = syclDeviceSynchronize();
  check_error_and_throw_sycl(error_sync2, "init_lock_arrays_sycl: post init");
}

template <typename T>
void finalize_lock_arrays_sycl() {
  if (SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;
  auto error_free1 = syclFree(SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h);
  check_error_and_throw_sycl(error_free1, "finalize_lock_arrays_sycl: free device locks");
  auto error_free2 = syclHostFree(SYCL_SPACE_ATOMIC_LOCKS_NODE_h);
  check_error_and_throw_sycl(error_free2, "finalize_lock_arrays_sycl: free host locks");
  SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
  SYCL_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;
#ifdef DESUL_SYCL_RDC
  DESUL_IMPL_COPY_SYCL_LOCK_ARRAYS_TO_DEVICE();
#endif
}

template void init_lock_arrays_sycl<int>();
template void finalize_lock_arrays_sycl<int>();

}  // namespace Impl

}  // namespace desul
#endif
