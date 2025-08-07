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

#ifndef KOKKOS_IMPL_ERROR_HPP
#define KOKKOS_IMPL_ERROR_HPP

#include <string>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Assert.hpp>
#include <Kokkos_Core_Export.h>

namespace Kokkos::Impl {

[[noreturn]] KOKKOSCORE_EXPORT void throw_runtime_exception(
    const std::string &msg);
[[noreturn]] void throw_bad_alloc(std::string_view memory_space_name,
                                  std::size_t size, std::string_view label);
KOKKOSCORE_EXPORT void log_warning(const std::string &msg);

std::string human_memory_size(size_t bytes);

}  // namespace Kokkos::Impl

#endif
