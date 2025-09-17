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

module;

#include <Kokkos_SIMD.hpp>

export module kokkos.simd_impl;

export {
  namespace Kokkos::Experimental {
  namespace simd_abi::Impl {
  using ::Kokkos::Experimental::simd_abi::Impl::native_abi;
  using ::Kokkos::Experimental::simd_abi::Impl::native_fixed_abi;
  }  // namespace simd_abi::Impl

  namespace Impl {
  using ::Kokkos::Experimental::Impl::abi_set;
  using ::Kokkos::Experimental::Impl::data_type_set;
  using ::Kokkos::Experimental::Impl::data_types;
  using ::Kokkos::Experimental::Impl::device_abi_set;
  using ::Kokkos::Experimental::Impl::host_abi_set;
  using ::Kokkos::Experimental::Impl::Identity;
  }  // namespace Impl
  }  // namespace Kokkos::Experimental
}
