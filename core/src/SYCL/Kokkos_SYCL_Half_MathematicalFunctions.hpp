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

#ifndef KOKKOS_SYCL_HALF_MATHEMATICALFUNCTIONS_HPP_
#define KOKKOS_SYCL_HALF_MATHEMATICALFUNCTIONS_HPP_

#ifdef KOKKOS_IMPL_SYCL_BHALF_TYPE_DEFINED

namespace Kokkos {

KOKKOS_INLINE_FUNCTION bool isnan(Kokkos::Experimental::bhalf_t x) {
  return sycl::ext::oneapi::experimental::isnan(
      Kokkos::Experimental::bhalf_t::impl_type(x));
}

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t fabs(
    Kokkos::Experimental::bhalf_t x) {
  return sycl::ext::oneapi::experimental::fabs(
      Kokkos::Experimental::bhalf_t::impl_type(x));
}

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t fmin(
    Kokkos::Experimental::bhalf_t x, Kokkos::Experimental::bhalf_t y) {
  using Kokkos::Experimental::bhalf_t;
  return sycl::ext::oneapi::experimental::fmin(bhalf_t::impl_type(x),
                                               bhalf_t::impl_type(y));
}

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t fmax(
    Kokkos::Experimental::bhalf_t x, Kokkos::Experimental::bhalf_t y) {
  using Kokkos::Experimental::bhalf_t;
  return sycl::ext::oneapi::experimental::fmax(bhalf_t::impl_type(x),
                                               bhalf_t::impl_type(y));
}

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t fma(
    Kokkos::Experimental::bhalf_t x, Kokkos::Experimental::bhalf_t y,
    Kokkos::Experimental::bhalf_t z) {
  using Kokkos::Experimental::bhalf_t;
  return sycl::ext::oneapi::experimental::fma(
      bhalf_t::impl_type(x), bhalf_t::impl_type(y), bhalf_t::impl_type(z));
}

}  // namespace Kokkos
#endif  // KOKKOS_IMPL_SYCL_BHALF_TYPE_DEFINED

#endif  // KOKKOS_SYCL_HALF_MATHEMATICALFUNCTIONS_HPP_
