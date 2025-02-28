/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_LOCK_BASED_FETCH_OP_SYCL_HPP_
#define DESUL_ATOMICS_LOCK_BASED_FETCH_OP_SYCL_HPP_

#include <desul/atomics/Common.hpp>
#include <desul/atomics/Lock_Array_SYCL.hpp>
#include <desul/atomics/Thread_Fence_SYCL.hpp>
#include <type_traits>

namespace desul {
namespace Impl {

template <class Oper,
          class T,
          class MemoryOrder,
          class MemoryScope,
          // equivalent to:
          //   requires !atomic_always_lock_free(sizeof(T))
          std::enable_if_t<!atomic_always_lock_free(sizeof(T)), int> = 0>
T device_atomic_fetch_oper(const Oper& op,
                           T* const dest,
                           dont_deduce_this_parameter_t<const T> val,
                           MemoryOrder /*order*/,
                           MemoryScope scope) {
	abort();
  return *dest;
}

template <class Oper,
          class T,
          class MemoryOrder,
          class MemoryScope,
          // equivalent to:
          //   requires !atomic_always_lock_free(sizeof(T))
          std::enable_if_t<!atomic_always_lock_free(sizeof(T)), int> = 0>
T device_atomic_oper_fetch(const Oper& op,
                           T* const dest,
                           dont_deduce_this_parameter_t<const T> val,
                           MemoryOrder /*order*/,
                           MemoryScope scope) {
          abort();
  return *dest;
}
}  // namespace Impl
}  // namespace desul

#endif
