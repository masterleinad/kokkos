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

template <typename... Args>
int printf(const char* format, Args... args) {
  return KOKKOS_IMPL_DO_NOT_USE_PRINTF(format, args...);
}

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_IMPL_PRINTF_HPP */
