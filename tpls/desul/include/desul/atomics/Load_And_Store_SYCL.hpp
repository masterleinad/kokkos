/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_LOAD_AND_STORE_SYCL_HPP_
#define DESUL_ATOMICS_LOAD_AND_STORE_SYCL_HPP_

#include <desul/atomics/Adapt_SYCL.hpp>
#if defined(__ACPP__)
#include <hipSYCL/sycl/libkernel/sscp/builtins/assert.hpp>
#endif
#include <desul/atomics/Lock_Free_Types_SYCL.hpp>

namespace desul {
namespace Impl {

template <class T, class MemoryOrder, class MemoryScope>
void device_atomic_store(T* ptr, T val, MemoryOrder, MemoryScope scope) {
  if constexpr (sizeof(T) == 4) {
    static_assert(sizeof(unsigned int) == 4,
                  "this function assumes an unsigned int is 32-bit");
    sycl_atomic_ref<unsigned int, MemoryOrder, MemoryScope> ref(
        reinterpret_cast<unsigned int&>(*ptr));
    ref.store(reinterpret_cast<unsigned int&>(val));
  } else if constexpr (sizeof(T) == 8) {
    static_assert(sizeof(unsigned long long int) == 8,
                  "this function assumes an unsigned long long is 64-bit");
    sycl_atomic_ref<unsigned long long, MemoryOrder, MemoryScope> ref(
        reinterpret_cast<unsigned long long&>(*ptr));
    ref.store(reinterpret_cast<unsigned long long&>(val));
  } else {
    // This is a way to avoid deadlock in a subgroup
    int done = 0;
#if defined(__ACPP__)
    // FIXME_ACPP The lock-based path for types that are not 4 or 8 bytes
    // needs the oneAPI sub-group extensions (this_sub_group, group_ballot,
    // sub_group_mask), which AdaptiveCpp does not provide. This cannot be a
    // static_assert: acpp compiles host and device in one pass and
    // DESUL_IF_ON_DEVICE is a runtime branch there, so host-only callers
    // (e.g. the HostThreadTeam work stealing on a 16-byte pair) instantiate
    // this function too. A kernel that actually reaches it reports a device
    // assertion (abort on the host device; on Metal the kernel continues).
    __acpp_if_target_device(__acpp_sscp_assert_fail(
        "desul: SYCL atomics on types that are not 4 or 8 bytes are not "
        "supported with AdaptiveCpp",
        __FILE__, __LINE__, __func__);)
#else
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20250000
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#else
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#endif
    using sycl::ext::oneapi::group_ballot;
    using sycl::ext::oneapi::sub_group_mask;
    sub_group_mask active = group_ballot(sg, 1);
    sub_group_mask done_active = group_ballot(sg, 0);
    while (active != done_active) {
      if (!done) {
        if (lock_address_sycl((void*)ptr, scope)) {
          if (std::is_same<MemoryOrder, MemoryOrderSeqCst>::value)
            atomic_thread_fence(MemoryOrderRelease(), scope);
          atomic_thread_fence(MemoryOrderAcquire(), scope);
          *ptr = val;
          unlock_address_sycl((void*)ptr, scope);
          done = 1;
        }
      }
      done_active = group_ballot(sg, done);
    }
#endif
  }
}

template <class T, class MemoryOrder, class MemoryScope>
T device_atomic_load(const T* ptr, MemoryOrder, MemoryScope scope) {
  if constexpr (sizeof(T) == 4) {
    static_assert(sizeof(unsigned int) == 4,
                  "this function assumes an unsigned int is 32-bit");
    sycl_atomic_ref<unsigned int, MemoryOrder, MemoryScope> ref(
        reinterpret_cast<unsigned int&>(const_cast<T&>(*ptr)));
    auto sycl_return = ref.load();
    return reinterpret_cast<T&>(sycl_return);
  } else if constexpr (sizeof(T) == 8) {
    static_assert(sizeof(unsigned long long int) == 8,
                  "this function assumes an unsigned long long is 64-bit");
    sycl_atomic_ref<unsigned long long, MemoryOrder, MemoryScope> ref(
        reinterpret_cast<unsigned long long&>(const_cast<T&>(*ptr)));
    auto sycl_return = ref.load();
    return reinterpret_cast<T&>(sycl_return);
  } else {
    // This is a way to avoid deadlock in a subgroup
    T ret;
    int done = 0;
#if defined(__ACPP__)
    // FIXME_ACPP The lock-based path for types that are not 4 or 8 bytes
    // needs the oneAPI sub-group extensions (this_sub_group, group_ballot,
    // sub_group_mask), which AdaptiveCpp does not provide. This cannot be a
    // static_assert: acpp compiles host and device in one pass and
    // DESUL_IF_ON_DEVICE is a runtime branch there, so host-only callers
    // (e.g. the HostThreadTeam work stealing on a 16-byte pair) instantiate
    // this function too. A kernel that actually reaches it reports a device
    // assertion (abort on the host device; on Metal the kernel continues).
    __acpp_if_target_device(__acpp_sscp_assert_fail(
        "desul: SYCL atomics on types that are not 4 or 8 bytes are not "
        "supported with AdaptiveCpp",
        __FILE__, __LINE__, __func__);)
#else
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20250000
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#else
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#endif
    using sycl::ext::oneapi::group_ballot;
    using sycl::ext::oneapi::sub_group_mask;
    sub_group_mask active = group_ballot(sg, 1);
    sub_group_mask done_active = group_ballot(sg, 0);
    while (active != done_active) {
      if (!done) {
        if (lock_address_sycl((void*)ptr, scope)) {
          if (std::is_same<MemoryOrder, MemoryOrderSeqCst>::value)
            atomic_thread_fence(MemoryOrderRelease(), scope);
          atomic_thread_fence(MemoryOrderAcquire(), scope);
          ret = *ptr;
          unlock_address_sycl((void*)ptr, scope);
          done = 1;
        }
      }
      done_active = group_ballot(sg, done);
    }
#endif
    return ret;
  }
}

}  // namespace Impl
}  // namespace desul

#endif
