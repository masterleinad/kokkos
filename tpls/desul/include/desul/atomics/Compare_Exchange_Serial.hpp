/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_THREAD_FENCE_SERIAL_HPP_
#define DESUL_ATOMICS_THREAD_FENCE_SERIAL_HPP_

#include <desul/atomics/Common.hpp>

namespace desul {
namespace Impl {

template <class T, class MemoryOrder, class MemoryScope>
T host_atomic_exchange(T* dest, T value, MemoryOrder, MemoryScope) {
 *dest=value;
return dest;
}

}  // namespace impl
}  // namespace desul

#endif
