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

#ifndef KOKKOS_IMPL_PRINTF_HPP
#define KOKKOS_IMPL_PRINTF_HPP

#include <Kokkos_Macros.hpp>

namespace Kokkos {

// We are forwarding the format string to printf but gcc and clang warn about
// the format string not being a literal.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif

template <typename... Args>
int printf(const char* format, Args... args) {
  return KOKKOS_IMPL_DO_NOT_USE_PRINTF(format, args...);
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_IMPL_PRINTF_HPP */
