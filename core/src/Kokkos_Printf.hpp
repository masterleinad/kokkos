//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_PRINTF_HPP
#define KOKKOS_PRINTF_HPP

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_SYCL
#include <sycl/sycl.hpp>
#else
#include <cstdio>
#endif

namespace Kokkos {

// In contrast to std::printf, return void to get a consistent behavior across
// backends. The GPU backends always return 1 and NVHPC only compiles if we
// don't ask for the return value.
#if defined(KOKKOS_ENABLE_OPENMPTARGET) && defined(KOKKOS_ARCH_INTEL_GPU)
using ::printf;
#else
template <typename... Args>
KOKKOS_FORCEINLINE_FUNCTION void printf(const char* format, Args... args) {
  if constexpr (sizeof...(Args) == 0)
    ::printf("%s", format);
  else
    ::printf(format, args...);
}
#endif

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_PRINTF_HPP */
